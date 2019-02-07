package sample
import java.math._
import scala.util._
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.gsl._


object samplers {

	def sampleCRT(rng: gsl.gsl_rng, m: Double, gammazero: Double): Double = {

		var(sum, bparam) = (0.0, 0.0)	
		for(i <- 0 to (m.toInt - 1)) {
			bparam = gammazero / (gammazero + i)
			if(gsl.gsl_rng_uniform(rng) <= bparam)
				sum += 1
		}
		return sum
	}

	def TruncPoisson(rng: gsl.gsl_rng, lam: Double): Double = {
		
		var lambda = lam
		var (m, pmf, prob) = (0, 1.0, 0.0)
		var rand = new Random()
		if (lambda >= 1) {
			while (m <= 0) {
				m = gsl.gsl_ran_poisson(rng, lambda)
			}
		}
		else {
			m = 1
			if (lambda <= 0.000001) {
				lambda = 0.000001
			}
			prob = math.pow(lambda,m)*math.exp(-lambda)/(m*(1-math.exp(-lambda)))
			while(prob/pmf <= gsl.gsl_rng_uniform(rng)) {
				pmf = pmf-prob
				m += 1
				prob = math.pow(lambda,m)*math.exp(-lambda)/(m*(1-math.exp(-lambda)))
			}
		}
		return m
	}

}
