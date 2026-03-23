package net.finmath.finitedifference.solvers;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;

/**
 * Shared matrix-free assembly utilities for 1D theta-method finite-difference solvers.
 *
 * <p>
 * This class provides the common numerical building blocks used by:
 * </p>
 * <ul>
 *   <li>{@link FDMThetaMethod1D}</li>
 *   <li>{@link FDMThetaMethod1DTwoState}</li>
 * </ul>
 *
 * <p>
 * It assembles the 1D spatial operator directly into tridiagonal coefficients,
 * without constructing dense matrices.
 * </p>
 * 
 * @author Alessandro Gnoatto
 */
public final class ThetaMethod1DAssembly {

	private static final double SAFE_TIME_EPSILON = 1E-6;

	private ThetaMethod1DAssembly() {
	}

	/**
	 * Container for model coefficients on a 1D grid.
	 */
	public static final class ModelCoefficients {
		private final double[] drift;
		private final double[] variance;
		private final double shortRate;

		public ModelCoefficients(final double[] drift, final double[] variance, final double shortRate) {
			this.drift = drift;
			this.variance = variance;
			this.shortRate = shortRate;
		}

		public double[] getDrift() {
			return drift;
		}

		public double[] getVariance() {
			return variance;
		}

		public double getShortRate() {
			return shortRate;
		}
	}

	private static final class RowCoefficients {
		private final double lower;
		private final double diag;
		private final double upper;

		private RowCoefficients(final double lower, final double diag, final double upper) {
			this.lower = lower;
			this.diag = diag;
			this.upper = upper;
		}
	}

	/**
	 * Evaluates the 1D drift, variance and short rate on the supplied grid at one time.
	 *
	 * @param model The finite-difference model.
	 * @param xGrid The 1D state grid.
	 * @param time The running time.
	 * @return The model coefficients.
	 */
	public static ModelCoefficients buildModelCoefficients(
			final FiniteDifferenceEquityModel model,
			final double[] xGrid,
			final double time) {

		final int n = xGrid.length;

		final double[] mu = new double[n];
		final double[] variance = new double[n];

		for(int i = 0; i < n; i++) {
			final double x = xGrid[i];

			mu[i] = model.getDrift(time, x)[0];

			final double[][] factorLoading = model.getFactorLoading(time, x);

			double localVariance = 0.0;
			for(int f = 0; f < factorLoading[0].length; f++) {
				final double b = factorLoading[0][f];
				localVariance += b * b;
			}
			variance[i] = localVariance;
		}

		return new ModelCoefficients(mu, variance, getShortRate(model, time));
	}

	/**
	 * Computes the continuously compounded short rate implied by the model discount curve.
	 *
	 * @param model The finite-difference model.
	 * @param time The running time.
	 * @return The short rate.
	 */
	public static double getShortRate(final FiniteDifferenceEquityModel model, final double time) {
		final double safeTime = time == 0.0 ? SAFE_TIME_EPSILON : Math.max(time, SAFE_TIME_EPSILON);
		return -Math.log(model.getRiskFreeCurve().getDiscountFactor(safeTime)) / safeTime;
	}

	/**
	 * Builds the left-hand side matrix for the theta step:
	 * {@code I - theta * dt * L(t_{m+1})}.
	 *
	 * @param lhs The tridiagonal matrix to overwrite.
	 * @param xGrid The 1D state grid.
	 * @param drift The drift values on the grid.
	 * @param variance The variance values on the grid.
	 * @param shortRate The short rate.
	 * @param deltaTau The time step size.
	 * @param theta The theta parameter.
	 */
	public static void buildThetaLeftHandSide(
			final TridiagonalMatrix lhs,
			final double[] xGrid,
			final double[] drift,
			final double[] variance,
			final double shortRate,
			final double deltaTau,
			final double theta) {

		final double alpha = theta * deltaTau;
		for(int i = 0; i < xGrid.length; i++) {
			final RowCoefficients spatial = spatialOperatorRow(i, xGrid, drift[i], variance[i], shortRate);
			lhs.lower[i] = -alpha * spatial.lower;
			lhs.diag[i] = 1.0 - alpha * spatial.diag;
			lhs.upper[i] = -alpha * spatial.upper;
		}
	}

	/**
	 * Builds the right-hand side operator for the theta step:
	 * {@code I + (1-theta) * dt * L(t_m)}.
	 *
	 * @param rhsOperator The tridiagonal matrix to overwrite.
	 * @param xGrid The 1D state grid.
	 * @param drift The drift values on the grid.
	 * @param variance The variance values on the grid.
	 * @param shortRate The short rate.
	 * @param deltaTau The time step size.
	 * @param theta The theta parameter.
	 */
	public static void buildThetaRightHandSide(
			final TridiagonalMatrix rhsOperator,
			final double[] xGrid,
			final double[] drift,
			final double[] variance,
			final double shortRate,
			final double deltaTau,
			final double theta) {

		final double alpha = (1.0 - theta) * deltaTau;
		for(int i = 0; i < xGrid.length; i++) {
			final RowCoefficients spatial = spatialOperatorRow(i, xGrid, drift[i], variance[i], shortRate);
			rhsOperator.lower[i] = alpha * spatial.lower;
			rhsOperator.diag[i] = 1.0 + alpha * spatial.diag;
			rhsOperator.upper[i] = alpha * spatial.upper;
		}
	}

	/**
	 * Applies a tridiagonal operator to a vector.
	 *
	 * @param matrix The tridiagonal matrix.
	 * @param vector The vector.
	 * @return {@code matrix * vector}.
	 */
	public static double[] apply(final TridiagonalMatrix matrix, final double[] vector) {
		final int n = vector.length;
		final double[] result = new double[n];

		for(int i = 0; i < n; i++) {
			double value = matrix.diag[i] * vector[i];
			if(i > 0) {
				value += matrix.lower[i] * vector[i - 1];
			}
			if(i < n - 1) {
				value += matrix.upper[i] * vector[i + 1];
			}
			result[i] = value;
		}

		return result;
	}

	/**
	 * Overwrites one row with a Dirichlet condition.
	 *
	 * @param matrix The matrix.
	 * @param rhs The right-hand side vector.
	 * @param row The row to overwrite.
	 * @param value The prescribed value.
	 */
	public static void overwriteAsDirichlet(
			final TridiagonalMatrix matrix,
			final double[] rhs,
			final int row,
			final double value) {

		matrix.lower[row] = 0.0;
		matrix.diag[row] = 1.0;
		matrix.upper[row] = 0.0;
		rhs[row] = value;
	}

	private static RowCoefficients spatialOperatorRow(
			final int i,
			final double[] x,
			final double mu,
			final double variance,
			final double r) {

		final int n = x.length;
		final double halfVariance = 0.5 * variance;

		double t1Lower = 0.0;
		double t1Diag = 0.0;
		double t1Upper = 0.0;

		double t2Lower = 0.0;
		double t2Diag = 0.0;
		double t2Upper = 0.0;

		if(i == 0) {
			final double h1 = x[1] - x[0];
			final double h2 = x[2] - x[1];

			t1Diag = -1.0 / h1;
			t1Upper = 1.0 / h1;

			t2Diag = -2.0 / (h1 * h2);
			t2Upper = 2.0 / (h1 * (h1 + h2));
		}
		else if(i == n - 1) {
			final double h0 = x[i] - x[i - 1];
			final double h3 = x[i - 1] - x[i - 2];

			t1Lower = -1.0 / h0;
			t1Diag = 1.0 / h0;

			t2Lower = 2.0 / (h0 * (h0 + h3));
			t2Diag = -2.0 / (h3 * h0);
		}
		else {
			final double h0 = x[i] - x[i - 1];
			final double h1 = x[i + 1] - x[i];

			t1Lower = -h1 / (h0 * (h1 + h0));
			t1Diag = (h1 - h0) / (h1 * h0);
			t1Upper = h0 / (h1 * (h0 + h1));

			t2Lower = 2.0 / (h0 * (h0 + h1));
			t2Diag = -2.0 / (h0 * h1);
			t2Upper = 2.0 / (h1 * (h0 + h1));
		}

		return new RowCoefficients(
				mu * t1Lower + halfVariance * t2Lower,
				mu * t1Diag + halfVariance * t2Diag - r,
				mu * t1Upper + halfVariance * t2Upper);
	}
}