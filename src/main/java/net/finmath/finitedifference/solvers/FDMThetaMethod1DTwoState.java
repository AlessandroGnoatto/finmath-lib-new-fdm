package net.finmath.finitedifference.solvers;

import java.util.function.DoubleUnaryOperator;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.BarrierOption;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;

/**
 * Direct two-state theta-method solver for 1D knock-in barrier options.
 *
 * <p>
 * Regime 0 = not yet activated (barrier not yet hit).
 * Regime 1 = already activated (barrier has been hit).
 * </p>
 *
 * <p>
 * This implementation is matrix-free:
 * </p>
 *
 * <ul>
 *   <li>The active regime is solved on the full grid using a tridiagonal theta step.</li>
 *   <li>The inactive regime is solved only on the continuation-side subgrid.</li>
 *   <li>On the already-hit region, the inactive regime is eliminated by imposing {@code inactive = active}.</li>
 * </ul>
 *
 * <p>
 * Currently supports only European exercise.
 * </p>
 */
public class FDMThetaMethod1DTwoState implements FDMSolver {

	private static final double EPSILON = 1E-10;

	private final FiniteDifferenceEquityModel model;
	private final BarrierOption product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final Exercise exercise;
	private final TwoStateActiveBoundaryProvider activeBoundaryProvider;

	public FDMThetaMethod1DTwoState(
			final FiniteDifferenceEquityModel model,
			final BarrierOption product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final TwoStateActiveBoundaryProvider activeBoundaryProvider) {
		this.model = model;
		this.product = product;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
		this.exercise = exercise;
		this.activeBoundaryProvider = activeBoundaryProvider;
	}

	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {

		if(!exercise.isEuropean()) {
			throw new IllegalArgumentException("FDMThetaMethod1DTwoState currently supports only European exercise.");
		}

		if(activeBoundaryProvider == null) {
			throw new IllegalArgumentException("Active boundary provider must not be null.");
		}

		final BarrierType barrierType = product.getBarrierType();
		if(barrierType != BarrierType.DOWN_IN && barrierType != BarrierType.UP_IN) {
			throw new IllegalArgumentException("FDMThetaMethod1DTwoState is only for knock-in barrier options.");
		}

		final double[] xGrid = spaceTimeDiscretization.getSpaceGrid(0).getGrid();
		final int nX = xGrid.length;
		if(nX < 2) {
			throw new IllegalArgumentException("Need at least two grid points.");
		}

		final int barrierIndex = findBarrierIndex(xGrid, product.getBarrierValue());
		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int numberOfTimeSteps = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		double[] inactive = new double[nX];
		double[] active = new double[nX];

		for(int i = 0; i < nX; i++) {
			final double x = xGrid[i];
			final double payoff = valueAtMaturity.applyAsDouble(x);

			active[i] = payoff;
			inactive[i] = isAlreadyHitRegion(x, barrierType, product.getBarrierValue()) ? payoff : product.getRebate();
		}

		final double[][] solutionSurface = new double[nX][timeLength];
		for(int i = 0; i < nX; i++) {
			solutionSurface[i][0] = inactive[i];
		}

		for(int m = 0; m < numberOfTimeSteps; m++) {

			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(numberOfTimeSteps - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(numberOfTimeSteps - (m + 1));

			final double tau_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);
			final double currentTime = spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tau_mp1;

			final double lowerActiveBoundary = activeBoundaryProvider.getLowerBoundaryValue(currentTime, xGrid[0]);
			final double upperActiveBoundary = activeBoundaryProvider.getUpperBoundaryValue(currentTime, xGrid[nX - 1]);

			final double[] nextActive = solveVanillaStep(
					xGrid,
					active,
					t_m,
					t_mp1,
					deltaTau,
					lowerActiveBoundary,
					upperActiveBoundary);

			final double[] nextInactive = new double[nX];

			switch(barrierType) {
			case DOWN_IN:
				fillDownInInactiveStep(
						xGrid,
						barrierIndex,
						inactive,
						active,
						nextActive,
						nextInactive,
						t_m,
						t_mp1,
						deltaTau,
						currentTime);
				break;

			case UP_IN:
				fillUpInInactiveStep(
						xGrid,
						barrierIndex,
						inactive,
						active,
						nextActive,
						nextInactive,
						t_m,
						t_mp1,
						deltaTau,
						currentTime);
				break;

			default:
				throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
			}

			active = nextActive;
			inactive = nextInactive;

			for(int i = 0; i < nX; i++) {
				solutionSurface[i][m + 1] = inactive[i];
			}
		}

		return solutionSurface;
	}

	@Override
	public double[] getValue(
			final double evaluationTime,
			final double time,
			final DoubleUnaryOperator valueAtMaturity) {

		final double[][] values = getValues(time, valueAtMaturity);
		final double tau = time - evaluationTime;
		final int timeIndex = spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);

		final double[] column = new double[values.length];
		for(int i = 0; i < values.length; i++) {
			column[i] = values[i][timeIndex];
		}
		return column;
	}

	private void fillDownInInactiveStep(
			final double[] xGrid,
			final int barrierIndex,
			final double[] inactivePrevious,
			final double[] activePrevious,
			final double[] activeNext,
			final double[] inactiveNext,
			final double t_m,
			final double t_mp1,
			final double deltaTau,
			final double currentTime) {

		for(int i = 0; i <= barrierIndex; i++) {
			inactiveNext[i] = activeNext[i];
		}

		if(barrierIndex == xGrid.length - 1) {
			return;
		}

		final double[] subGrid = sliceGrid(xGrid, barrierIndex, xGrid.length - 1);
		final double[] previousSub = new double[subGrid.length];

		for(int j = 0; j < subGrid.length; j++) {
			previousSub[j] = inactivePrevious[barrierIndex + j];
		}

		previousSub[0] = activePrevious[barrierIndex];

		final double discountedNoHitValue = getDiscountedNoHitValue(currentTime);

		final double[] nextSub = solveVanillaStep(
				subGrid,
				previousSub,
				t_m,
				t_mp1,
				deltaTau,
				activeNext[barrierIndex],
				discountedNoHitValue);

		for(int j = 0; j < nextSub.length; j++) {
			inactiveNext[barrierIndex + j] = nextSub[j];
		}

		for(int i = 0; i <= barrierIndex; i++) {
			inactiveNext[i] = activeNext[i];
		}
	}

	private void fillUpInInactiveStep(
			final double[] xGrid,
			final int barrierIndex,
			final double[] inactivePrevious,
			final double[] activePrevious,
			final double[] activeNext,
			final double[] inactiveNext,
			final double t_m,
			final double t_mp1,
			final double deltaTau,
			final double currentTime) {

		for(int i = barrierIndex; i < xGrid.length; i++) {
			inactiveNext[i] = activeNext[i];
		}

		if(barrierIndex == 0) {
			return;
		}

		final double[] subGrid = sliceGrid(xGrid, 0, barrierIndex);
		final double[] previousSub = new double[subGrid.length];

		for(int j = 0; j < subGrid.length; j++) {
			previousSub[j] = inactivePrevious[j];
		}

		previousSub[subGrid.length - 1] = activePrevious[barrierIndex];

		final double discountedNoHitValue = getDiscountedNoHitValue(currentTime);

		final double[] nextSub = solveVanillaStep(
				subGrid,
				previousSub,
				t_m,
				t_mp1,
				deltaTau,
				discountedNoHitValue,
				activeNext[barrierIndex]);

		for(int j = 0; j < nextSub.length; j++) {
			inactiveNext[j] = nextSub[j];
		}

		for(int i = barrierIndex; i < xGrid.length; i++) {
			inactiveNext[i] = activeNext[i];
		}
	}

	private double[] solveVanillaStep(
			final double[] xGrid,
			final double[] previousValues,
			final double t_m,
			final double t_mp1,
			final double deltaTau,
			final double lowerBoundaryValue,
			final double upperBoundaryValue) {

		final int n = xGrid.length;

		if(n != previousValues.length) {
			throw new IllegalArgumentException("Grid and solution vector size mismatch.");
		}

		if(n == 1) {
			return new double[] { lowerBoundaryValue };
		}

		if(n == 2) {
			return new double[] { lowerBoundaryValue, upperBoundaryValue };
		}

		final double theta = spaceTimeDiscretization.getTheta();

		final double tSafe_m = Math.max(t_m, 1E-6);
		final double tSafe_mp1 = Math.max(t_mp1, 1E-6);

		final double r_m = -Math.log(model.getRiskFreeCurve().getDiscountFactor(tSafe_m)) / tSafe_m;
		final double r_mp1 = -Math.log(model.getRiskFreeCurve().getDiscountFactor(tSafe_mp1)) / tSafe_mp1;

		final double[] mu_m = new double[n];
		final double[] mu_mp1 = new double[n];
		final double[] a_m = new double[n];
		final double[] a_mp1 = new double[n];

		for(int i = 0; i < n; i++) {
			final double x = xGrid[i];

			mu_m[i] = model.getDrift(t_m, x)[0];
			mu_mp1[i] = model.getDrift(t_mp1, x)[0];

			final double[][] b_m = model.getFactorLoading(t_m, x);
			final double[][] b_mp1 = model.getFactorLoading(t_mp1, x);

			double variance_m = 0.0;
			for(int f = 0; f < b_m[0].length; f++) {
				final double b = b_m[0][f];
				variance_m += b * b;
			}

			double variance_mp1 = 0.0;
			for(int f = 0; f < b_mp1[0].length; f++) {
				final double b = b_mp1[0][f];
				variance_mp1 += b * b;
			}

			a_m[i] = variance_m;
			a_mp1[i] = variance_mp1;
		}

		final TridiagonalMatrix lhs = new TridiagonalMatrix(n);
		final TridiagonalMatrix rhsOperator = new TridiagonalMatrix(n);

		buildThetaLeftHandSide(lhs, xGrid, mu_mp1, a_mp1, r_mp1, deltaTau, theta);
		buildThetaRightHandSide(rhsOperator, xGrid, mu_m, a_m, r_m, deltaTau, theta);

		final double[] rhs = apply(rhsOperator, previousValues);

		overwriteAsDirichlet(lhs, rhs, 0, lowerBoundaryValue);
		overwriteAsDirichlet(lhs, rhs, n - 1, upperBoundaryValue);

		final double[] next = ThomasSolver.solve(lhs.lower, lhs.diag, lhs.upper, rhs);
		next[0] = lowerBoundaryValue;
		next[n - 1] = upperBoundaryValue;

		return next;
	}

	private void buildThetaLeftHandSide(
			final TridiagonalMatrix lhs,
			final double[] xGrid,
			final double[] mu,
			final double[] a,
			final double r,
			final double deltaTau,
			final double theta) {

		final double alpha = theta * deltaTau;
		for(int i = 0; i < xGrid.length; i++) {
			final RowCoefficients spatial = spatialOperatorRow(i, xGrid, mu[i], a[i], r);
			lhs.lower[i] = -alpha * spatial.lower;
			lhs.diag[i] = 1.0 - alpha * spatial.diag;
			lhs.upper[i] = -alpha * spatial.upper;
		}
	}

	private void buildThetaRightHandSide(
			final TridiagonalMatrix rhsOperator,
			final double[] xGrid,
			final double[] mu,
			final double[] a,
			final double r,
			final double deltaTau,
			final double theta) {

		final double alpha = (1.0 - theta) * deltaTau;
		for(int i = 0; i < xGrid.length; i++) {
			final RowCoefficients spatial = spatialOperatorRow(i, xGrid, mu[i], a[i], r);
			rhsOperator.lower[i] = alpha * spatial.lower;
			rhsOperator.diag[i] = 1.0 + alpha * spatial.diag;
			rhsOperator.upper[i] = alpha * spatial.upper;
		}
	}

	private RowCoefficients spatialOperatorRow(
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

	private double[] apply(final TridiagonalMatrix matrix, final double[] vector) {
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

	private void overwriteAsDirichlet(
			final TridiagonalMatrix lhs,
			final double[] rhs,
			final int row,
			final double value) {

		lhs.lower[row] = 0.0;
		lhs.diag[row] = 1.0;
		lhs.upper[row] = 0.0;
		rhs[row] = value;
	}

	private double getDiscountedNoHitValue(final double currentTime) {

		if(product.getRebate() == 0.0) {
			return 0.0;
		}

		final double t = Math.max(currentTime, EPSILON);
		final double maturity = product.getMaturity();

		if(t >= maturity) {
			return product.getRebate();
		}

		final double discountFactorAtCurrentTime = model.getRiskFreeCurve().getDiscountFactor(t);
		final double discountFactorAtMaturity = model.getRiskFreeCurve().getDiscountFactor(maturity);

		return product.getRebate() * discountFactorAtMaturity / discountFactorAtCurrentTime;
	}

	private int findBarrierIndex(final double[] grid, final double barrier) {
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - barrier) < 1E-12) {
				return i;
			}
		}

		throw new IllegalArgumentException(
				"Barrier must coincide with a 1D grid node for direct two-state knock-in pricing.");
	}

	private double[] sliceGrid(final double[] grid, final int startInclusive, final int endInclusive) {
		final double[] result = new double[endInclusive - startInclusive + 1];
		for(int i = 0; i < result.length; i++) {
			result[i] = grid[startInclusive + i];
		}
		return result;
	}

	private boolean isAlreadyHitRegion(
			final double x,
			final BarrierType barrierType,
			final double barrier) {

		switch(barrierType) {
		case DOWN_IN:
			return x <= barrier;
		case UP_IN:
			return x >= barrier;
		default:
			throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
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
}