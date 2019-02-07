package jgppf
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
// --------------------  J-GPPF  --------------------
// --------------------------------------------------

class JointModel(rng: gsl.gsl_rng, data: PoissonData, epsilonlc: Double, kB: Integer, kY: Integer, outDir: String) {
	
	var KB = kB
	var KY = kY
	var N = data.N
	var D = data.D
	var V = data.V
	var epsilon = epsilonlc
	var outDirectory = outDir

	// initialize hyperparameters
	// for network
	var(azeroB, bzeroB, czeroB, dzeroB) = (1.0, 1.0, sqrt(N.toDouble), 1.0)
	var(ezeroB, fzeroB, gzeroB) = (1.0, 1.0, 1.0)
	var(hzeroB, mzeroB, nzeroB, szeroB) = (1.0, 1.0, 1.0, 1.0)
	var(tzeroB, cB, gammaB, xiB) = (1.0, 1.0, 1.0, 0.1)
	// for corpus
	var(azeroY, bzeroY, czeroY, dzeroY) = (1.0, 1.0, sqrt(D.toDouble), 1.0)
	var(ezeroY, fzeroY, gzeroY) = (1.0, 1.0, 1.0)
	var(hzeroY, mzeroY, nzeroY, szeroY) = (1.0, 1.0, 1.0, 1.0)
	var(tzeroY, cY, gammaY, xiY) = (1.0, 1.0, 1.0, 0.1)

	// initialize matrices and arrays
	// for network
	var phink	= Array.fill[Double](N,KB)(1.0/KB)
	var phinss	= Array.fill[Double](N)(1.0)		// sum(phink,1)
	var phikss	= Array.fill[Double](KB)(1.0*N/KB)	// sum(phink,0)
	var phikss2	= Array.fill[Double](KB)(1.0*N/(KB*KB))	// sum(phink^2,0)
	var rkB		= Array.fill[Double](KB)(1.0*gammaB/KB)
	var akB		= Array.fill[Double](KB)(1.0)
	var cn		= Array.fill[Double](N)(1.0)
	var psiwk	= Array.fill[Double](V,KB)(1.0/V)
	var xndotk	= Array.fill[Double](N,KB)(0.0)
	var ydotndotk	= Array.fill[Double](N,KB)(0.0)
	var xk		= Array.fill[Double](KB)(0.0)
	// for corpus
	var thetadk	= Array.fill[Double](D,KY)(1.0/KY)
	var thetadss	= Array.fill[Double](D)(1.0)		// sum(thetadk,1)
	var thetakss	= Array.fill[Double](KY)(1.0*D/KY)	// sum(thetadk,0)
	var rkY		= Array.fill[Double](KY)(1.0*gammaY/KY)
	var akY		= Array.fill[Double](KY)(1.0)
	var cd		= Array.fill[Double](D)(1.0)
	var betawk	= Array.fill[Double](V,KY)(1.0/V)
	var ydotwk	= Array.fill[Double](V,(KY+KB))(0.0)
	var yddotk	= Array.fill[Double](D,KY)(0.0)
	var yk		= Array.fill[Double](KY+KB)(0.0)
	var uk		= Array.fill[Double](KY)(0.0)
	// for Z
	var Zphikss	= Array.fill[Double](KB)(0.0)
	// for debugging and display
	var phinkss	= Array.fill[Double](N,KB)(0.0)
	var psiwkss	= Array.fill[Double](V,KB)(0.0)
	var rkBss	= Array.fill[Double](KB)(0.0)
	var thetadkss	= Array.fill[Double](D,KY)(0.0)
	var betawkss	= Array.fill[Double](V,KY)(0.0)
	var rkYss	= Array.fill[Double](KY)(0.0)

	def construct() {
		// Initialize Zphikss
		for (n <- 0 to (N-1)) {
			for (k <- 0 to (KB-1)) {
				var training_Ndsz_n = data.Ndsz(n) // remove heldout Y links from this (if applicable)
				if (data.mSY > 0) {
					// subtract the missing Y links
					var d = 0
					for (zbin <- 0 to (data.Z.row_ids(n).size-1)) {
						d = data.Z.row_ids(n)(zbin)
						for (nz <- 0 to (data.mY.row_ids(d).size-1)) {
							training_Ndsz_n -= psiwk(data.mY.row_ids(d)(nz))(k)
						}
					}
				}
				Zphikss(k) += training_Ndsz_n * phink(n)(k)
			}
		}
	}
	construct

	def train(BurninITER: Int, CollectionITER: Int, data: PoissonData, Option: Int, netOption: Int, samplePeriod: Int) {
		
		var(param1, param2, xiBparam1, xiBparam2, xiYparam1, xiYparam2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
		var(rkBsum, rkYsum, lsum1, logpsum1, lsum2, logpsum2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
		var(sk, akBsum, akYsum, epsilonparam1, epsilonparam2) = (0.0, 0.0, 0.0, 0.0, 0.0)
		var(training_phikss_k, training_Ndsz_n, training_betakss_k) = (0.0, 0.0, 0.0)
		var(kBcount, m, n, w, d, value) = (0, 0, 0, 0, 0, 0)
		
		akBsum = akB.sum
		akYsum = akY.sum
		
		// Gibbs sampling iterations start
		for (i <- 0 to (BurninITER+CollectionITER-1)) {

			// interim output
			if (i==0 || (i+1)%100==0) {
				print("Iteration: " + (i+1) + "\n")
			}
			if (i % samplePeriod == 0) {
				printSamples(i)
			}
			
			// reset a few statistics first; O(NK)
			xndotk = Array.fill[Double](N,KB)(0.0)
			xk = Array.fill[Double](KB)(0.0)
			ydotndotk = Array.fill[Double](N,KB)(0.0)
			yddotk = Array.fill[Double](D,KY)(0.0)
			ydotwk = Array.fill[Double](V,(KY+KB))(0.0)
			yk = Array.fill[Double](KY+KB)(0.0)
			epsilonparam1 = 0.0
			epsilonparam2 = 0.0

			// sampling starts
			// network sampling of latent counts; O(SK)
			for (n <- 0 to (N-1)) {
				for (nz <- 0 to (data.B.row_ids(n).size-1)) {
					m = data.B.row_ids(n)(nz)
					value = data.B.row_counts(n)(nz)
					var countsample = Array.fill[Int](KB)(0)
					var pmf = Array.fill[Double](KB)(0.0)
					for (k <- 0 to (KB-1)) {
						pmf(k) = mathutilities.mathutils.minguard(rkB(k)*phink(n)(k)*phink(m)(k))
					}
					// normalization
					var normsum = pmf.sum
					for (k <- 0 to (KB-1)) {
						pmf(k) = pmf(k) / normsum
					}
					if (netOption == 0) {
						// binary network, use truncated poisson to estimate "value"
						value = sample.samplers.TruncPoisson(rng, normsum.toInt).toInt
					}
					gsl.gsl_ran_multinomial(rng, KB.toLong, value, pmf, countsample)
					// update sufficient statistics of x
					for (k <- 0 to (KB-1)) {
						xndotk(n)(k) += countsample(k)
						xndotk(m)(k) += countsample(k)
						xk(k) += countsample(k)
					}
				}
			}
			
			// corpus sampling of latent counts; O(SK)
			for (d <- 0 to (D-1)) {
				var KBcount = KB * data.Dnsz(d).toInt
				for (nz <- 0 to (data.Y.row_ids(d).size-1)) {
					w = data.Y.row_ids(d)(nz)
					value = data.Y.row_counts(d)(nz)
					var countsample = Array.fill[Int](KBcount+KY)(0)
					var pmf = Array.fill[Double](KBcount+KY)(0)
					kBcount = 0
					if (Option == 1) {
						for (k <- 0 to (KB-1)) {
							// for network groups
							for (zbin <- 0 to (data.Zt.row_ids(d).size-1)) {
								n = data.Zt.row_ids(d)(zbin)
								pmf(kBcount) = mathutilities.mathutils.minguard(epsilon*rkB(k)*phink(n)(k)*psiwk(w)(k))
								kBcount += 1
							}
						}
					}
					for (k <- 0 to (KY-1)) {
						// for count data related groups
						pmf(KBcount+k) = mathutilities.mathutils.minguard(rkY(k)*thetadk(d)(k)*betawk(w)(k))
					}
					// normalization
					var normsum = pmf.sum
					for (k <- 0 to (KBcount+KY-1)) {
						pmf(k) = pmf(k) / normsum
					}
					gsl.gsl_ran_multinomial(rng, (KBcount+KY).toLong, value, pmf, countsample)
					// update sufficient statistics of Y
					kBcount = 0
					for (k <- 0 to (KB-1)) {
						// for network groups
						for (zbin <- 0 to (data.Zt.row_ids(d).size-1)) {
							n = data.Zt.row_ids(d)(zbin)
							ydotndotk(n)(k) += countsample(kBcount)
							ydotwk(w)(k) += countsample(kBcount)
							yk(k) += countsample(kBcount)
							epsilonparam1 += countsample(kBcount)
							kBcount += 1
						}
					}
					for (k <- 0 to (KY-1)) {
						yddotk(d)(k) += countsample(KBcount+k)
						ydotwk(w)(KB+k) += countsample(KBcount+k)
						yk(KB+k) += countsample(KBcount+k)
					}
				}
			}

			// network sampling of phink and cn; O(NK+N)
			for (n <- 0 to (N-1)) {
				// reset the statistics about phi
				phinss(n) = 0.0
				for (k <- 0 to (KB-1)) {
					// sampling of phink
					phikss(k) -= phink(n)(k) // avoids recomputing sum_n(phink)
					phikss2(k) -= pow(phink(n)(k),2) // avoids recomputing sum_n(phink^2)
					training_phikss_k = phikss(k) // remove heldout B links from this (if applicable)
					training_Ndsz_n = data.Ndsz(n) // remove heldout Y links from this (if applicable)
					if (data.mSN > 0) {
						// subtract the missing B links
						for (nz <- 0 to (data.mB.row_ids(n).size-1)) {
							training_phikss_k -= phink(data.mB.row_ids(n)(nz))(k)
						}
					}
					if (data.mSY > 0) {
						// subtract the missing Y links
						for (zbin <- 0 to (data.Z.row_ids(n).size-1)) {
							d = data.Z.row_ids(n)(zbin)
							for (nz <- 0 to (data.mY.row_ids(d).size-1)) {
								training_Ndsz_n -= psiwk(data.mY.row_ids(d)(nz))(k)
							}
						}
					}
					Zphikss(k) -= training_Ndsz_n * phink(n)(k)
					param1 = azeroB + xndotk(n)(k) + ydotndotk(n)(k)
					if (Option == 1) {
						param2 = 1.0 / (cn(n) + rkB(k) * (training_phikss_k + epsilon*training_Ndsz_n))
					}
					else {
						param2 = 1.0 / (cn(n) + rkB(k) * training_phikss_k)
					}
					phink(n)(k) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
					if (i >= BurninITER) {
						phinkss(n)(k) += phink(n)(k) / CollectionITER
					}
					// update sufficient statistics for phi
					phinss(n) += phink(n)(k)
					phikss(k) += phink(n)(k)
					phikss2(k) += pow(phink(n)(k), 2)
					Zphikss(k) += training_Ndsz_n * phink(n)(k)
				}
				// sampling of cn
				param1 = czeroB + KB*azeroB
				param2 = 1.0 / (dzeroB + phinss(n))
				cn(n) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
			}

			// network sampling of rk, lk, and gammak; O(3*K)
			rkBsum = 0.0
			lsum1 = 0.0
			logpsum1 = 0.0
			xiBparam1 = 0.0
			xiBparam2 = 0.0
			for (k <- 0 to (KB-1)) {
				// sample rkB
				sk = (pow(phikss(k), 2) - phikss2(k)) / 2.0
				if (data.mSN > 0) {
					// subtract the missing B links
					for (n <- 0 to (N-1)) {
						for (nz <- 0 to (data.mB.row_ids(n).size-1)) {
							sk -= phink(n)(k) * phink(data.mB.row_ids(n)(nz))(k)
						}
					}
				}
				// note: Zphikss has already been compensated for missing Y links, no need to subtract Z*phi*psi terms
				param1 = 1.0*gammaB/KB + xk(k) + yk(k)
				if (Option == 1) {
					param2 = 1.0 / (cB + sk + epsilon*Zphikss(k))
				}
				else {
					param2 = 1.0 / (cB + sk)
				}
				rkB(k) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
				rkBsum += rkB(k)
				if (i >= BurninITER) {
					rkBss(k) += rkB(k) / CollectionITER
				}
				// sample lk's for the updates of gammaB
				lsum1 += sample.samplers.sampleCRT(rng, yk(k), 1.0*gammaB/KB)
				if (Option == 1) {
					logpsum1 += mathutilities.mathutils.logguard(1.0 + (sk + epsilon*Zphikss(k))/cB)
				}
				else {
					logpsum1 += mathutilities.mathutils.logguard(1.0 + 1.0*sk/cB)
				}
				epsilonparam2 += rkB(k) * Zphikss(k)
				// sample psiwk only for the joint model, no need to sample for the disjoint model
				if (Option == 1) {
					// sample psiwk
					var pmf = Array.fill[Double](V)(0.0)
					var alpha_k = Array.fill[Double](V)(0.0)
					for (w <- 0 to (V-1)) {
						// get the parameters for the Dirichlet distribution
						alpha_k(w) = 1.0*xiB + ydotwk(w)(k)
					}
					gsl.gsl_ran_dirichlet(rng, V, alpha_k, pmf)
					for (w <- 0 to (V-1)) {
						psiwk(w)(k) = mathutilities.mathutils.minguard(pmf(w))
						if (i >= BurninITER) {
							psiwkss(w)(k) += psiwk(w)(k) / CollectionITER
						}
						// sample the CRT random variables for the updates of the Dirichlet hyperparameter
						xiBparam1 += sample.samplers.sampleCRT(rng, ydotwk(w)(k), xiB)
					}
					// sample the beta random variables for the updates of the Dirichlet hyperparameter
					xiBparam2 += mathutilities.mathutils.logguard(1.0 - gsl.gsl_ran_beta(rng, yk(k), V*xiB))
				}
			}
			// sample gammaB
			param1 = ezeroB + lsum1
			param2 = 1.0 / (fzeroB + 1.0*logpsum1/KB)
			gammaB = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))

			// corpus sampling of rk, lk, gammak, thetadk, and betawk; O(DK+VK+K)
			lsum2 = 0.0
			logpsum2 = 0.0
			rkYsum = 0.0
			xiYparam1 = 0.0
			xiYparam2 = 0.0
			thetadss = Array.fill[Double](D)(0.0)
			for (k <- 0 to (KY-1)) {
				// sample thetadk
				thetakss(k) = 0.0
				lsum1 = 0.0
				logpsum1 = 0.0
				for (d <- 0 to (D-1)) {
					param1 = azeroY + yddotk(d)(k)
					training_betakss_k = 1.0 // remove missing Y links from this (if applicable)
					if (data.mSY > 0) {
						// subtract the missing Y links
						for (nz <- 0 to (data.mY.row_ids(d).size-1)) {
							training_betakss_k -= betawk(data.mY.row_ids(d)(nz))(k)
						}
					}
					param2 = 1.0 / (cd(d) + rkY(k)*training_betakss_k)
					thetadk(d)(k) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
					thetakss(k) += thetadk(d)(k)
					thetadss(d) += thetadk(d)(k)
					if (i >= BurninITER) {
						thetadkss(d)(k) += thetadk(d)(k) / CollectionITER
					}
					// sample ldky's for the updates of akY's
					lsum1 += sample.samplers.sampleCRT(rng, yddotk(d)(k), akY(k))
					logpsum1 += mathutilities.mathutils.logguard(1.0 + rkY(k)/cd(d))
				}
				// sample betawk
				var pmf = Array.fill[Double](V)(0.0)
				var alpha_k = Array.fill[Double](V)(0.0)
				for (w <- 0 to (V-1)) {
					// get the parameters for the Dirichlet distribution
					alpha_k(w) = 1.0*xiY + ydotwk(w)(k+KB)
				}
				gsl.gsl_ran_dirichlet(rng, V, alpha_k, pmf)
				for (w <- 0 to (V-1)) {
					betawk(w)(k) = mathutilities.mathutils.minguard(pmf(w))
					if (i >= BurninITER) {
						betawkss(w)(k) += betawk(w)(k) / CollectionITER
					}
					// sample the CRT random variables for the updates of the Dirichlet hyperparameter
					xiYparam1 += sample.samplers.sampleCRT(rng, ydotwk(w)(k+KB), xiY)
				}
				// prepare sufficient statistics for updating rkY, and compensate for missing Y links
				uk(k) = thetakss(k)
				if (data.mSY > 0) {
					for (d <- 0 to (D-1)) {
						for (nz <- 0 to (data.mY.row_ids(d).size-1)) {
							uk(k) -= thetadk(d)(k) * betawk(data.mY.row_ids(d)(nz))(k)
						}
					}
				}
				// sample the beta random variables for the updates of the Dirichlet hyperparameter
				xiYparam2 += mathutilities.mathutils.logguard(1.0 - gsl.gsl_ran_beta(rng, yk(k+KB), V*xiY))
				// sample rkY
				param1 = 1.0*gammaY/KY + yk(k+KB)
				param2 = 1.0 / (cY + uk(k))
				rkY(k) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
				rkYsum += rkY(k)
				if (i >= BurninITER) {
					rkYss(k) += rkY(k) / CollectionITER
				}
				// sample lk's for the updates of gammaY
				lsum2 += sample.samplers.sampleCRT(rng, yk(k), 1.0*gammaY/KY)
				logpsum2 += mathutilities.mathutils.logguard(1.0 + uk(k)/cY)
			}
			// sample gammaY
			param1 = ezeroY + lsum2
			param2 = 1.0 / (fzeroY + 1.0*logpsum2/KY)
			gammaY = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))

			// corpus sampling of cd; O(D)
			for (d <- 0 to (D-1)) {
				param1 = czeroY + KY*azeroY
				param2 = 1.0 / (dzeroY + thetadss(d))
				cd(d) = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
			}

			// sample global variable
			param1 = gzeroB + gammaB
			param2 = 1.0 / (hzeroB + rkBsum)
			cB = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
			param1 = gzeroY + gammaY
			param2 = 1.0 / (hzeroY + rkYsum)
			cY = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))

			// sample xiB
			//param1 = szeroB + xiBparam1
			//param2 = 1.0 / (tzeroB - V*xiBparam2)
			//xiB = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
			// sample xiY
			//param1 = szeroY + xiYparam1
			//param2 = 1.0 / (tzeroY - V*xiYparam2)
			//xiY = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
			// sample epsilon
			//param1 = szeroY + epsilonparam1
			//param2 = 1.0 / (tzeroY + epsilonparam2)
			//epsilon = mathutilities.mathutils.minguard(gsl.gsl_ran_gamma(rng, param1, param2))
		}
		// end of Gibbs sampling iteration loop
	}

	
	def printResults() {
		// saves the final expected values
		// first create a new directory
		val resultDirectoryName = outDirectory + "/JGPPF/expectedValues"
		val resultDirectory = new File(resultDirectoryName)
		resultDirectory.mkdirs()

		// save rkY
		printutilities.printutils.printVec(rkYss, resultDirectoryName + "/rkY.txt", KY)

		// save thetadk
		printutilities.printutils.printMat(thetadkss, resultDirectoryName + "/thetadk.txt", KY, D)

		// save betawk
		printutilities.printutils.printMat(betawkss, resultDirectoryName + "/betawk.txt", KY, V)

		// save psiwk
		printutilities.printutils.printMat(psiwkss, resultDirectoryName + "/psiwk.txt", KB, V)

		// save rkB
		printutilities.printutils.printVec(rkBss, resultDirectoryName + "/rkB.txt", KB)

		// save phink
		printutilities.printutils.printMat(phinkss, resultDirectoryName + "/phink.txt", KB, N)
	}

	def printSamples(iter: Int) {
		// saves interim samples
		// first create a directory, if not there
		val iterDirectoryName = outDirectory + "/JGPPF/iterations"
		val iterDirectory = new File(iterDirectoryName)
		iterDirectory.mkdirs()

		// save rkY
		printutilities.printutils.printVec(rkY, iterDirectoryName + "/rkY-itr%04d.txt".format(iter), KY)

		// save thetadk
		printutilities.printutils.printMat(thetadk, iterDirectoryName + "/thetadk-itr%04d.txt".format(iter), KY, D)

		// save betawk
		printutilities.printutils.printMat(betawk, iterDirectoryName + "/betawk-itr%04d.txt".format(iter), KY, V)

		// save psiwk
		printutilities.printutils.printMat(psiwk, iterDirectoryName + "/psiwk-itr%04d.txt".format(iter), KB, V)

		// save rkB
		printutilities.printutils.printVec(rkB, iterDirectoryName + "/rkB-itr%04d.txt".format(iter), KB)

		// save phink
		printutilities.printutils.printMat(phink, iterDirectoryName + "/phink-itr%04d.txt".format(iter), KB, N)
	}

	def generateSample(Option: Int, netOption: Int) {
		// generates random B,Y samples from the trained matrices, and saves to file
		// first create a directory, if not there
		val genDirectoryName = outDirectory + "/JGPPF/generatedSamples"
		val genDirectory = new File(genDirectoryName)
		genDirectory.mkdirs()

		// generate B
		var Bnew = Array.fill[Integer](N,N)(0)
		var(numEntries, x, y) = (0, 0, 0)
		var lambda = 0.0
		for (n <- 0 to (N-1)) {
			for (m <- (n+1) to (N-1)) {
				lambda = 0.0
				for (k <- 0 to (KB-1)) {
					lambda += rkBss(k) * phinkss(n)(k) * phinkss(m)(k)
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

		// generate Y
		var Ynew = Array.fill[Integer](D,V)(0)
		numEntries = 0
		for (d <- 0 to (D-1)) {
			for (w <- 0 to (V-1)) {
				lambda = 0.0
				for (k <- 0 to (KY-1)) {
					lambda += rkYss(k) * thetadkss(d)(k) * betawkss(w)(k)
				}
				if (Option == 1) {
					// sampling from joint model
					for (k <- 0 to (KB-1)) {
						for (zbin <- 0 to (data.Zt.row_ids(d).size-1)) {
							lambda += epsilon * rkBss(k) * phinkss(data.Zt.row_ids(d)(zbin))(k) * psiwkss(w)(k)
						}
					}
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

