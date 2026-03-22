package net.finmath.finitedifference.solvers;

public final class ThomasSolver {

	private ThomasSolver() {
	}

	public static double[] solve(
			final double[] lower,
			final double[] diag,
			final double[] upper,
			final double[] rhs) {

		final int n = diag.length;

		final double[] cPrime = new double[n];
		final double[] dPrime = new double[n];
		final double[] x = new double[n];

		cPrime[0] = upper[0] / diag[0];
		dPrime[0] = rhs[0] / diag[0];

		for(int i = 1; i < n; i++) {
			final double denom = diag[i] - lower[i] * cPrime[i - 1];
			cPrime[i] = i < n - 1 ? upper[i] / denom : 0.0;
			dPrime[i] = (rhs[i] - lower[i] * dPrime[i - 1]) / denom;
		}

		x[n - 1] = dPrime[n - 1];
		for(int i = n - 2; i >= 0; i--) {
			x[i] = dPrime[i] - cPrime[i] * x[i + 1];
		}

		return x;
	}
}