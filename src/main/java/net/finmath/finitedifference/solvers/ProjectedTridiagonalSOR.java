package net.finmath.finitedifference.solvers;

/**
 * Projected SOR specialized to tridiagonal linear complementarity problems.
 *
 * <pre>
 *     A x >= b
 *     x >= obstacle
 *     (A x - b)_i (x_i - obstacle_i) = 0
 * </pre>
 */
public final class ProjectedTridiagonalSOR {

	private ProjectedTridiagonalSOR() {
	}

	public static double[] solve(
			final TridiagonalMatrix matrix,
			final double[] rhs,
			final double[] obstacle,
			final double[] initialGuess,
			final double omega,
			final int maxIterations,
			final double tolerance) {

		return solve(
				matrix.lower,
				matrix.diag,
				matrix.upper,
				rhs,
				obstacle,
				initialGuess,
				omega,
				maxIterations,
				tolerance);
	}

	public static double[] solve(
			final double[] lower,
			final double[] diag,
			final double[] upper,
			final double[] rhs,
			final double[] obstacle,
			final double[] initialGuess,
			final double omega,
			final int maxIterations,
			final double tolerance) {

		validateInputs(lower, diag, upper, rhs, obstacle, initialGuess);

		final int n = diag.length;
		final double[] x = initialGuess.clone();

		for(int i = 0; i < n; i++) {
			x[i] = Math.max(x[i], obstacle[i]);
		}

		for(int iter = 0; iter < maxIterations; iter++) {
			double maxChange = 0.0;

			for(int i = 0; i < n; i++) {
				final double aii = diag[i];
				if(aii == 0.0) {
					throw new ArithmeticException("Zero diagonal entry encountered at i=" + i);
				}

				final double left = i > 0 ? lower[i] * x[i - 1] : 0.0;
				final double right = i < n - 1 ? upper[i] * x[i + 1] : 0.0;

				final double gaussSeidelValue = (rhs[i] - left - right) / aii;
				final double relaxedValue = (1.0 - omega) * x[i] + omega * gaussSeidelValue;
				final double projectedValue = Math.max(obstacle[i], relaxedValue);

				maxChange = Math.max(maxChange, Math.abs(projectedValue - x[i]));
				x[i] = projectedValue;
			}

			if(tolerance > 0.0 && maxChange <= tolerance) {
				break;
			}
		}

		return x;
	}

	public static double complementarityResidualInfNorm(
			final TridiagonalMatrix matrix,
			final double[] rhs,
			final double[] obstacle,
			final double[] x) {

		return complementarityResidualInfNorm(
				matrix.lower,
				matrix.diag,
				matrix.upper,
				rhs,
				obstacle,
				x);
	}

	public static double complementarityResidualInfNorm(
			final double[] lower,
			final double[] diag,
			final double[] upper,
			final double[] rhs,
			final double[] obstacle,
			final double[] x) {

		validateInputs(lower, diag, upper, rhs, obstacle, x);

		double maxResidual = 0.0;
		for(int i = 0; i < diag.length; i++) {
			final double ax =
					(i > 0 ? lower[i] * x[i - 1] : 0.0)
					+ diag[i] * x[i]
					+ (i < diag.length - 1 ? upper[i] * x[i + 1] : 0.0);

			final double primalViolation = Math.max(0.0, obstacle[i] - x[i]);
			final double dualViolation = Math.max(0.0, rhs[i] - ax);
			final double slack = x[i] - obstacle[i];
			final double residual = Math.max(
					Math.max(primalViolation, dualViolation),
					Math.abs(slack * (ax - rhs[i])));

			maxResidual = Math.max(maxResidual, residual);
		}

		return maxResidual;
	}

	private static void validateInputs(
			final double[] lower,
			final double[] diag,
			final double[] upper,
			final double[] rhs,
			final double[] obstacle,
			final double[] initialGuess) {

		if(lower == null || diag == null || upper == null || rhs == null || obstacle == null || initialGuess == null) {
			throw new IllegalArgumentException("Input arrays must not be null.");
		}
		if(diag.length == 0) {
			throw new IllegalArgumentException("System dimension must be positive.");
		}
		if(lower.length != diag.length
				|| upper.length != diag.length
				|| rhs.length != diag.length
				|| obstacle.length != diag.length
				|| initialGuess.length != diag.length) {
			throw new IllegalArgumentException("All arrays must have the same length.");
		}
	}
}