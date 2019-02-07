package cgppf
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
// --------------------  C-GPPF  --------------------
// --------------------------------------------------

class CorpusModel(rng: gsl.gsl_rng, data: PoissonData, numTopics: Integer, outDir: String) {

	var D = data.D
	var V = data.V
	var K = numTopics
	var outDirectory = outDir

	// initialize hyperparameters
	var(azero, bzero, czero, dzero, ezero) = (1.0, 1.0, 1.0, 1.0, 1.0)
	var(fzero, gzero, hzero, c) = (1.0, sqrt(D.toDouble), 1.0, 1.0)
	var(mzero, nzero, szero, tzero) = (1.0, 1.0, 1.0, 1.0)

	// initialize data structures
	var thetadk	= Array.fill[Double](D,K)(1.0/K)
	var betawk	= Array.fill[Double](V,K)(1.0/K)
	var thetakss	= Array.fill[Double](K)(0.0)
	var betakss	= Array.fill[Double](K)(0.0)
	var thetadss	= Array.fill[Double](D)(0.0)
	var betawss	= Array.fill[Double](V)(0.0)
	var gammak	= Array.fill[Double](K)(1.0*azero/bzero)
	var lk		= Array.fill[Double](K)(0.0)
	var rk		= Array.fill[Double](K)(1.0*gammak(0)/c)
	var uk		= Array.fill[Double](K)(0.0)
	var cd		= Array.fill[Double](D)(1.0*gzero/hzero)
	var ad		= Array.fill[Double](D)(1.0*ezero/fzero)
	var lddot	= Array.fill[Double](D)(0.0)
	var sw		= Array.fill[Double](V)(1.0*szero/tzero)
	var bw		= Array.fill[Double](V)(1.0*mzero/nzero)
	var lwdot	= Array.fill[Double](V)(0.0)
	var xdotwk	= Array.fill[Double](V,K)(0.0)
	var xddotk	= Array.fill[Double](D,K)(0.0)
	var xk		= Array.fill[Double](K)(0.0)
	var pdsum	= Array.fill[Double](D)(0.0)
	var qwsum	= Array.fill[Double](V)(0.0)

	// for debugging and display
	var thetadkss	= Array.fill[Double](D,K)(0.0)
	var betawkss	= Array.fill[Double](V,K)(0.0)
	var rkss	= Array.fill[Double](K)(0.0)

	def construct() {
    
		for (k <- 0 to (K-1)) {
			for (d <- 0 to (D-1)) {
				thetadss(d) += thetadk(d)(k) // sum over topics for each different document
				thetakss(k) += thetadk(d)(k) // sum over documents for each different topic
			}
			for (w <- 0 to (V-1)) {
				betawss(w) += betawk(w)(k)    // sum over topics for each different word
				betakss(k) += betawk(w)(k)    // sum over words for each different topic
			}
		}

	}
	construct

	def train(BurninITER: Int, CollectionITER: Int, data: PoissonData, samplePeriod: Int) {

		var(rsum, gammasum, param1, param2) = (0.0, 0.0, 0.0, 0.0)
		var(w, value, training_betakss_k, training_thetakss_k) = (0, 0, 0.0, 0.0)

		// temporary structures for getting word ids and their counts
		for (i <- 0 to (CollectionITER.toInt + BurninITER- 1)) {
			if (i==0 || (i+1)%200 == 0) {
				print("Iteration: " + (i+1) + "\n");
			}
			if (i % samplePeriod == 0) {
				printSamples(i)
			}
			
			// reset statistics
			rsum = 0.0
			gammasum = 0.0

			for (k <- 0 to (K-1)) {
				xk(k) = 0.0
				for (d <- 0 to (D-1)) {
					xddotk(d)(k) = 0.0
				}
				for (w <- 0 to (V-1)) {
					xdotwk(w)(k) = 0.0
				}
			}
	
			// sampling starts
			// sampling of latent counts; O(SK)
			for (d <- 0 to (D-1)) {
				for (nz <- 0 to (data.Y.row_ids(d).size-1)) {
					w = data.Y.row_ids(d)(nz)
					value = data.Y.row_counts(d)(nz)
					var countsample = Array.fill[Int](K)(0)
					var pmf = Array.fill[Double](K)(0.0)
					for (k <- 0 to (K-1)) {
						pmf(k) = mathutilities.mathutils.minguard(rk(k)*thetadk(d)(k)*betawk(w)(k))
					}
					// normalization
					var normsum = pmf.sum
					for (k <- 0 to (K-1)) {
						pmf(k) = pmf(k) / normsum
					}
					gsl.gsl_ran_multinomial(rng, K.toLong, value, pmf, countsample);
					// update sufficient statistics of x
					for (k <- 0 to K-1) {
						xddotk(d)(k) += countsample(k)
						xdotwk(w)(k) += countsample(k)
						xk(k) += countsample(k)	
					}
				}
			} 
			
			// sampling of rk, lk, gammak, thetadk and betawk; O(DK+VK+K)
			for (k <- 0 to (K-1)) {
				thetakss(k) = 0.0 // reset the old statistics about theta
				for (d <- 0 to D-1) {
					training_betakss_k = betakss(k)
					if (data.mSY > 0) {
						// subtract the missing Y links
						for (nz <- 0 to (data.mY.row_ids(d).size-1)) {
							training_betakss_k -= betawk(data.mY.row_ids(d)(nz))(k)
						}
					}
					param1 = ad(d) + xddotk(d)(k)
					param2 = 1.0 / (cd(d) + rk(k) * training_betakss_k)
					pdsum(d) += mathutilities.mathutils.logguard(cd(d) * param2)
					// sample thetadk
					thetadk(d)(k) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
					if (i >= BurninITER) {
						thetadkss(d)(k) += thetadk(d)(k) / CollectionITER
					}
					// sample ldk
					lddot(d) += sample.samplers.sampleCRT(rng, xddotk(d)(k), ad(d))
					// update sufficient statistics for theta
					thetakss(k) += thetadk(d)(k)
					thetadss(d) += thetadk(d)(k)
				}
				betakss(k) = 0.0 // reset the old statistics about beta
				uk(k) = 0.0 // reset the sufficient statistics to be used for updating rk's
				for (w <- 0 to V - 1) {
					training_thetakss_k = thetakss(k)
					if (data.mSY > 0) {
						// subtract the missing Y links
						for (nz <- 0 to (data.mYt.row_ids(w).size-1)) {
							training_thetakss_k -= thetadk(data.mYt.row_ids(w)(nz))(k)
						}
					}
					param1 = bw(w) + xdotwk(w)(k)
					param2 = 1.0 / (sw(w) + rk(k) * training_thetakss_k)
					qwsum(w) += mathutilities.mathutils.logguard(sw(w) * param2)
					// sample betawk
					betawk(w)(k) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
					if (i >= BurninITER) {
						betawkss(w)(k) += betawk(w)(k) / CollectionITER
					}
					// sample lwk
					lwdot(w) += sample.samplers.sampleCRT(rng, xdotwk(w)(k), bw(w))
					// update sufficient statistics for beta
					betakss(k) += betawk(w)(k)
					betawss(w) += betawk(w)(k)
					// update sufficient statistics to be used for updating rk's
					uk(k) += betawk(w)(k) * training_thetakss_k
				}
				// sample rk
				param1 = gammak(k) + xk(k)
				param2 = 1.0 / (c + uk(k))
				rk(k) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
				if (i >= BurninITER) {
					rkss(k) += rk(k) / CollectionITER
				}
				rsum += rk(k)
				// sample lk
				lk(k) = sample.samplers.sampleCRT(rng, xk(k), gammak(k))
				// sample gammak
				param1 = azero + lk(k)
				param2 = 1.0 / (bzero - mathutilities.mathutils.logguard(c/(c+uk(k))))
				gammak(k) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
				gammasum += gammak(k)
			}	
        
			// sample ad, cd; O(D)
			for (d <- 0 to D -1 ) {
				// sample cd
				param1 = gzero + K*(ad(d))
				param2 = 1.0 / (hzero + thetadss(d))
				cd(d) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
				// sample ad (this was commented out in the c++ version)
				//param1 = ezero + lddot(d)
				//param2 = 1.0/(fzero - pdsum(d))			
				// *(ad+d) = minguard(gsl_ran_gamma(rng,param1,param2));			
				thetadss(d) = 0.0 // reset thetadss for next iteration
				lddot(d) = 0.0 // reset lddot for next iteration
				pdsum(d) = 0.0
			}
		
			// sample bw, sw; O(V)
			for (w <- 0 to V - 1) {
				// sample sw
				param1 = szero + K*(bw(w))
				param2 = 1.0 / (tzero + betawss(w))
				sw(w) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
				// sample bw (this was commented out in the c++ version)		
				//param1 = mzero + lwdot(w)
				//param2 = 1.0/(nzero - qwsum(w))			
				// *(bw+w) = minguard(gsl_ran_gamma(rng,param1,param2))
				betawss(w) = 0.0 // reset betawss for next iteration
				lwdot(w) = 0.0 // reset lwdot for next iteration
				qwsum(w) = 0.0 
			}

			// sample global variable (commenting this out to be consistent with N-GPPF)
			//c = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, (czero + gammasum), 1.0/(dzero + rsum)))
		}
		// end of Gibbs sampling loop
	}

	def printResults() {
		// saves the final expected values
		// first create a new directory
		val resultDirectoryName = outDirectory + "/CGPPF/expectedValues"
		val resultDirectory = new File(resultDirectoryName)
		resultDirectory.mkdirs()

		// save rkY
		printutilities.printutils.printVec(rkss, resultDirectoryName + "/rkY.txt", K)

		// save thetadk
		printutilities.printutils.printMat(thetadkss, resultDirectoryName + "/thetadk.txt", K, D)

		// save betawk
		printutilities.printutils.printMat(betawkss, resultDirectoryName + "/betawk.txt", K, V)
	}

	def printSamples(iter: Int) {
		// saves interim samples
		// first create a directory, if not there
		val iterDirectoryName = outDirectory + "/CGPPF/iterations"
		val iterDirectory = new File(iterDirectoryName)
		iterDirectory.mkdirs()

		// save rkY
		printutilities.printutils.printVec(rk, iterDirectoryName + "/rkY-itr%04d.txt".format(iter), K)

		// save thetadk
		printutilities.printutils.printMat(thetadk, iterDirectoryName + "/thetadk-itr%04d.txt".format(iter), K, D)

		// save betawk
		printutilities.printutils.printMat(betawk, iterDirectoryName + "/betawk-itr%04d.txt".format(iter), K, V)
	}

	def generateSample() {
		// generates random Y sample from the trained matrices, and saves to file
		// first create a directory, if not there
		val genDirectoryName = outDirectory + "/CGPPF/generatedSamples"
		val genDirectory = new File(genDirectoryName)
		genDirectory.mkdirs()

		// generate Y
		var Ynew = Array.fill[Integer](D,V)(0)
		var(numEntries, y) = (0, 0)
		var lambda = 0.0
		for (d <- 0 to (D-1)) {
			for (w <- 0 to (V-1)) {
				lambda = 0.0
				for (k <- 0 to (K-1)) {
					lambda += rkss(k) * thetadkss(d)(k) * betawkss(w)(k)
				}
				y = gsl.gsl_ran_poisson(rng, lambda)
				if (y > 0) {
					numEntries += 1
					Ynew(d)(w) = y
				}
			}
		}

		// save generated Y
		printutilities.printutils.printTrFile(Ynew, genDirectoryName + "/genY.txt", D, V, D + "\t" + V + "\t" + numEntries)
	}

}
