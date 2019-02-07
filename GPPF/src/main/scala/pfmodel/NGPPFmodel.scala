package ngppf
import scala.io.Source
import data._
import scala.collection.mutable.ListBuffer
import java.io._
import mathutilities._
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.gsl._
import sample._
import scala.collection.mutable.ArrayBuffer
import scala.math.sqrt
import scala.math.pow
import printutilities._

// --------------------------------------------------
// --------------------  N-GPPF  --------------------
// --------------------------------------------------

class NetworkModel(rng: gsl.gsl_rng, data: PoissonData, numTopics: Integer, outDir: String) {
	
	var N = data.N
	var K = numTopics
	var outDirectory = outDir

	// initialize hyperparameters
	var(azero, bzero, czero, dzero, ezero) = (1.0, 1.0, 1.0, 1.0, 1.0)
	var(fzero, gzero, hzero, c) = (1.0, sqrt(N.toDouble), 1.0, 1.0)

	// initialize data structures
	var phink 	= Array.fill[Double](N,K)(1.0/K)
	var lnk 	= Array.fill[Double](N,K)(0.0)
	var phinss 	= Array.fill[Double](N)(0.0)
	var phikss 	= Array.fill[Double](K)(0.0)
	var phikss2 	= Array.fill[Double](K)(0.0)
	var lk 		= Array.fill[Double](K)(0.0)
	var gammak 	= Array.fill[Double](K)(1.0*azero/bzero)
	var rk 		= Array.fill[Double](K)(1.0*gammak(0)/c)
	var an 		= Array.fill[Double](N)(1.0*ezero/fzero)
	var cn 		= Array.fill[Double](N)(1.0*gzero/hzero)
	var xndotk 	= Array.fill[Double](N,K)(0.0)
	var xk 		= Array.fill[Double](K)(0.0)

	// for debugging and display
	var phinkss 	= Array.fill[Double](N,K)(0.0)
	var rkss 	= Array.fill[Double](K)(0.0)

	def construct() {
		for (k <- 0 to (K-1)) {
			for (n <- 0 to (N-1)) {
				phikss(k) += phink(n)(k)
				phikss2(k) += pow(phink(n)(k),2)
			}
		}
	}
	construct

	def train(BurninITER: Int, CollectionITER: Int, data: PoissonData, netOption: Int, samplePeriod: Int) {
		
		var(param1, param2, training_phikss_k) = (0.0, 0.0, 0.0)
		var(gammasum, rsum, lnsum, pnsum) = (0.0, 0.0, 0.0, 0.0)
		var(value, m, c, sk) = (0, 0, 0.0, 0.0)
		
		// Gibbs sampling iteration starts
		for (i <- 0 to (BurninITER+CollectionITER-1)) {
			if (i==0 || (i+1)%200 == 0) {
				print("Iteration: " + (i+1) + "\n")
			}
			if (i % samplePeriod == 0) {
				printSamples(i)
			}
			
			// reset a few statistics first; O(NK)
			rsum = 0.0
			gammasum = 0.0
			for (k <- 0 to (K-1)) {
				xk(k) = 0.0
				for (n <- 0 to (N-1)) {
					xndotk(n)(k) = 0.0
				}
			}

			// sampling starts
			// sampling of latent counts; O(SK)
			for (n <- 0 to (N-2)) {
				for (nz <- 0 to (data.B.row_ids(n).size-1)) {
					m = data.B.row_ids(n)(nz)
					value = data.B.row_counts(n)(nz)
					var countsample = Array.fill[Int](K)(0)
					var pmf = Array.fill[Double](K)(0.0)
					for (k <- 0 to (K-1)) {
						pmf(k) = mathutilities.mathutils.minguard(rk(k)*phink(n)(k)*phink(m)(k))
					}
					// normalization
					var normsum = pmf.sum
					for (k <- 0 to (K-1)) {
						pmf(k) = pmf(k) / normsum
					}
					if (netOption == 0) {
						// binary network, use truncated poisson to estimate "value"
						value = sample.samplers.TruncPoisson(rng, normsum.toInt).toInt
					}
					gsl.gsl_ran_multinomial(rng, K.toLong, value, pmf, countsample)
					// update sufficient statistics of x
					for (k <- 0 to (K-1)) {
						xndotk(n)(k) += countsample(k)
						xndotk(m)(k) += countsample(k)
						xk(k) += countsample(k)
					}
				}
			}

			// sampling of rk, lk, gammak; O(3*K)
			for (k <- 0 to (K-1)) {
				sk = (pow(phikss(k),2) - phikss2(k))/2
				if (data.mSN > 0) {
					// subtract the missing B links
					for (n <- 0 to (N-1)) {
						for (nz <- 0 to (data.mB.row_ids(n).size-1)) {
							sk -= phink(n)(k) * phink(data.mB.row_ids(n)(nz))(k)
						}
					}
				}
				// sample rk
				param1 = gammak(k) + xk(k)
				param2 = 1.0 / (c + sk)
				rk(k) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
				if (i >= BurninITER) {
					rkss(k) += rk(k) / CollectionITER
				}
				rsum += rk(k)
				// sample lk
				lk(k) = sample.samplers.sampleCRT(rng, xk(k), gammak(k))
				// sample gammak
				param1 = azero + lk(k)
				param2 = 1.0 / (bzero - mathutilities.mathutils.logguard(c/(c+sk)))
				gammak(k) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
				gammasum += gammak(k)
			}

			// sampling of phink and cn; O(NK+N)
			for (n <- 0 to (N-1)) {
				// reset the statistics about phi
				phinss(n) = 0.0
				lnsum = 0.0
				pnsum = 0.0
				for (k <- 0 to (K-1)) {
					phikss(k) -= phink(n)(k) // avoids recomputing sum_n(phink)
					phikss2(k) -= pow(phink(n)(k),2) // avoids recomputing sum_n(phink^2)
					training_phikss_k = phikss(k) // remove heldout links from this (if applicable)
					if (data.mSN > 0) {
						// subtract the missing B links
						for (nz <- 0 to (data.mB.row_ids(n).size-1)) {
							training_phikss_k -= phink(data.mB.row_ids(n)(nz))(k)
						}
					}
					param1 = an(n) + xndotk(n)(k)
					param2 = 1.0 / (cn(n) + rk(k) * training_phikss_k) // this is 1/(c_n + r_k*s_nk)
					pnsum += mathutilities.mathutils.logguard(cn(n) * param2)
					// sample phink
					phink(n)(k) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
					if (i >= BurninITER) {
						phinkss(n)(k) += phink(n)(k) / CollectionITER
					}
					// sample lnk
					lnk(n)(k) = sample.samplers.sampleCRT(rng, xndotk(n)(k), an(n))
					lnsum += lnk(n)(k)
					// update sufficient statistics for phi
					phinss(n) += phink(n)(k)
					phikss(k) += phink(n)(k)
					phikss2(k) += pow(phink(n)(k),2)
				}
				// sample an (this was commented out in the c++ version)
				//param1 = ezero + lnsum
				//param2 = 1.0 / (fzero - pnsum)
				//an(n) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2)
				// sample cn
				param1 = gzero + K*an(n)
				param2 = 1.0 / (hzero + phinss(n))
				cn(n) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
			}
			// sample global variable (this is also commented out in the c++ version, and is listed as a hyperparameter)
			//c = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, czero+gammasum, 1.0/(dzero+rsum)))
		}
		// end of Gibbs sampling loop
	}

	
	def printResults() {
		// saves the final expected values
		// first create a new directory
		val resultDirectoryName = outDirectory + "/NGPPF/expectedValues"
		val resultDirectory = new File(resultDirectoryName)
		resultDirectory.mkdirs()

		// save rkB
		printutilities.printutils.printVec(rkss, resultDirectoryName + "/rkB.txt", K)

		// save phink
		printutilities.printutils.printMat(phinkss, resultDirectoryName + "/phink.txt", K, N)
	}

	def printSamples(iter: Int) {
		// saves interim samples
		// first create a directory, if not there
		val iterDirectoryName = outDirectory + "/NGPPF/iterations"
		val iterDirectory = new File(iterDirectoryName)
		iterDirectory.mkdirs()

		// save rkB
		printutilities.printutils.printVec(rk, iterDirectoryName + "/rkB-itr%04d.txt".format(iter), K)

		// save phink
		printutilities.printutils.printMat(phink, iterDirectoryName + "/phink-itr%04d.txt".format(iter), K, N)
	}

	def generateSample(netOption: Int) {
		// generates random B sample from the trained matrices, and saves to file
		// first create a directory, if not there
		val genDirectoryName = outDirectory + "/NGPPF/generatedSamples"
		val genDirectory = new File(genDirectoryName)
		genDirectory.mkdirs()

		// generate B
		var Bnew = Array.fill[Integer](N,N)(0)
		var(numEntries, x) = (0, 0)
		var lambda = 0.0
		for (n <- 0 to (N-1)) {
			for (m <- (n+1) to (N-1)) {
				lambda = 0.0
				for (k <- 0 to (K-1)) {
					lambda += rkss(k) * phinkss(n)(k) * phinkss(m)(k)
				}
				x = gsl.gsl_ran_poisson(rng, lambda)
				if (netOption == 0) {
					// binary network, truncate x
					if (x > 1) {x = 1}
				}
				if (x > 0) {
					numEntries += 1
					Bnew(n)(m) = x
				}
			}
		}

		// save generated B
		printutilities.printutils.printTrFile(Bnew, genDirectoryName + "/genB.txt", N, N, N + "\t" + numEntries)
	}

}

