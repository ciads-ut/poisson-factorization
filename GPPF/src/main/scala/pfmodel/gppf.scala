import scala.util.Random._
import data._
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.gsl._
import jgppf._
import ngppf._
import cgppf._

object gppf {
	def main(args: Array[String]) {

		// start of timer
		var t0 = System.currentTimeMillis
		var Temp = gsl.gsl_rng_default
		var rng = gsl.gsl_rng_alloc(Temp)
		var aRand = scala.util.Random

		// the seed for the rng	
		var random = aRand.nextInt(32767) % 10 + 1
		gsl.gsl_rng_set(rng, random)

		// load the config file
		var configFileName = "poisson.config"	// default config file
		if (args.size > 0) {
			configFileName = args(0)	// user-defined config file
		}
		println("Loading parameters: " + configFileName)
		var config = new Config(configFileName)

		// filenames
		var trFile1 = config.variables("NETWORK_TRAIN")		// training filename (network)
		var trFile2 = config.variables("CORPUS_TRAIN")		// training filename (corpus)
		var trZFile = config.variables("AUTHORS_TRAIN")		// training filename (authorship)
		var predFile1 = config.variables("NETWORK_HELDOUT")	// prediction filename (network)
		var predFile2 = config.variables("CORPUS_HELDOUT")	// prediction filename (corpus)

		// output directory
		var outDir = config.variables("OUT_DIR")
		
		// number of topics
		var KB = config.variables("NETWORK_TOPICS").toInt	// number of latent factors for B
		var KY = config.variables("CORPUS_TOPICS").toInt	// number of latent factors for Y
		
		// iterations
		var burnin = config.variables("BURNIN_ITER").toInt		// number of burnin iterations
		var collection = config.variables("COLLECT_ITER").toInt		// number of collection iterations

		// options
		var jointOption = 1	// 0: disjoint model, 1: joint model
		var epsilon = config.variables("EPSILON").toDouble		// mixing parameter between network and corpus
		var netOption = config.variables("COUNT_FLAG").toInt		// 0: binary network, 1: count network
		var samplePeriod = config.variables("OUTPUT_ITER").toInt	// save interim samples every 'samplePeriod' iterations

		// commands
		var modelSelection = config.variables("RUN_MODEL")
		var generateSamples = config.variables("GENERATE_SAMPLES").toInt

		// get data and train model
		if (modelSelection == "jgppf") {
			var dat = new PoissonData(trFile1, trFile2, trZFile, predFile1, predFile2)
			println("Running J-GPPF")
			var mod = new JointModel(rng, dat, epsilon, KB, KY, outDir)
			mod.train(burnin, collection, dat, jointOption, netOption, samplePeriod)
			mod.printResults()
			if (generateSamples == 1) {
				mod.generateSample(jointOption, netOption)
			}
		}
		else if (modelSelection == "ngppf") {
			var dat = new PoissonData(trFile1, "", "", predFile1, "")
			println("Running N-GPPF")
			var mod = new NetworkModel(rng, dat, KB, outDir)
			mod.train(burnin, collection, dat, netOption, samplePeriod)
			mod.printResults()
			if (generateSamples == 1) {
				mod.generateSample(netOption)
			}
		}
		else if (modelSelection == "cgppf") {
			var dat = new PoissonData("", trFile2, "", "", predFile2)
			println("Running C-GPPF")
			var mod = new CorpusModel(rng, dat, KY, outDir)
			mod.train(burnin, collection, dat, samplePeriod)
			mod.printResults()
			if (generateSamples == 1) {
				mod.generateSample()
			}
		}
		else {
			println("Error: unknown model selection \"RUN_MODEL\"")
		}
		
		//end timer and print elapsed time
		var elapsed_time = System.currentTimeMillis - t0
		var elapsed_time_minutes = (elapsed_time / (1000*60)).toInt
		var elapsed_time_seconds = (elapsed_time / (1000)).toInt
		if (elapsed_time_minutes > 5) {
			println("Elapsed time: " + elapsed_time_minutes + " minutes")
		}
		else {
			println("Elapsed time: " + elapsed_time_seconds + " seconds")
		}
	}	
}
