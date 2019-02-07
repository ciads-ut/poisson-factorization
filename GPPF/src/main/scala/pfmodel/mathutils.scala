package mathutilities
import java.math._


object mathutils {

	val LOWLIMIT = 1e-15
	
	def logguard(m : Double) : Double = {

		// provides guard against log(0)
		if (m < LOWLIMIT) {
			return math.log(LOWLIMIT)
		}
		else {
			return math.log(m)
		}
	}

	def minguard(m : Double) : Double = {

		// provides guard against number lower than LOWLIMIT
		if (m < LOWLIMIT) {
        		return (LOWLIMIT)
		}
    		else {
			return m
		}
	}
}
