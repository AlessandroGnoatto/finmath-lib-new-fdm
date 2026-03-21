package net.finmath.finitedifference.solvers.adi;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.products.BarrierOption;
import net.finmath.finitedifference.solvers.TwoStateActiveBoundaryProvider2D;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;

/**
 * Direct two-state ADI solver for 2D Heston knock-in barrier options.
 *
 * <p>
 * Regime 0 = inactive / not yet hit.
 * Regime 1 = active / already hit.
 * </p>
 *
 * <p>
 * This implementation mirrors the stabilized vanilla Heston ADI solver:
 * </p>
 * <ul>
 *   <li>Douglas-style ADI,</li>
 *   <li>two half-substeps per PDE time step,</li>
 *   <li>same corrected directional line matrices from {@link HestonADIStencilBuilder}.</li>
 * </ul>
 *
 * <p>
 * Two-state logic is added only via overlays:
 * </p>
 * <ul>
 *   <li>active outer boundaries from the injected provider,</li>
 *   <li>inactive continuation-side boundary = discounted no-hit value,</li>
 *   <li>inactive = active on the whole already-hit region in spot.</li>
 * </ul>
 *
 * <p>
 * Flattening convention:
 * {@code k = iS + iV * nS}, where {@code iS} is the fastest index.
 * </p>
 */
public class FDMHestonADI2DTwoState {

	private static final double EPSILON = 1E-10;

	private final FDMHestonModel model;
	private final BarrierOption product;
	private final Exercise exercise;
	private final TwoStateActiveBoundaryProvider2D activeBoundaryProvider;

	private final double theta;

	private final double[] sGrid;
	private final double[] vGrid;

	private final int nS;
	private final int nV;
	private final int n;

	private final HestonADIStencilBuilder stencilBuilder;

	public FDMHestonADI2DTwoState(
			final FDMHestonModel model,
			final BarrierOption product,
			final net.finmath.finitedifference.grids.SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final TwoStateActiveBoundaryProvider2D activeBoundaryProvider) {

		this.model = model;
		this.product = product;
		this.exercise = exercise;
		this.activeBoundaryProvider = activeBoundaryProvider;

		if(!exercise.isEuropean()) {
			throw new IllegalArgumentException("FDMHestonADI2DTwoState currently supports only European exercise.");
		}
		if(activeBoundaryProvider == null) {
			throw new IllegalArgumentException("Active boundary provider must not be null.");
		}
		if(product.getBarrierType() != BarrierType.DOWN_IN && product.getBarrierType() != BarrierType.UP_IN) {
			throw new IllegalArgumentException("FDMHestonADI2DTwoState is only for knock-in barrier options.");
		}

		this.theta = Math.max(0.5, spaceTimeDiscretization.getTheta());

		this.sGrid = spaceTimeDiscretization.getSpaceGrid(0).getGrid();
		this.vGrid = spaceTimeDiscretization.getSpaceGrid(1).getGrid();

		this.nS = sGrid.length;
		this.nV = vGrid.length;
		this.n = nS * nV;

		validateBarrierOnGridNode(sGrid, product.getBarrierValue());

		this.stencilBuilder = new HestonADIStencilBuilder(model, sGrid, vGrid);
	}

	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {
		return getValues(time, (s, v) -> valueAtMaturity.applyAsDouble(s));
	}

	public double[][] getValues(final double time, final DoubleBinaryOperator valueAtMaturity) {

		final int timeLength = model.getSpaceTimeDiscretization().getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int numberOfTimeSteps = model.getSpaceTimeDiscretization().getTimeDiscretization().getNumberOfTimeSteps();

		double[] uInactive = new double[n];
		double[] uActive = new double[n];

		initializeTerminalStates(uInactive, uActive, valueAtMaturity);

		applyTwoStateConditions(time, uInactive, uActive);

		final RealMatrix solutionSurface = new Array2DRowRealMatrix(n, timeLength);
		solutionSurface.setColumn(0, uInactive.clone());

		for(int m = 0; m < numberOfTimeSteps; m++) {
			final double dt = model.getSpaceTimeDiscretization().getTimeDiscretization().getTimeStep(m);
			final double currentTime =
					model.getSpaceTimeDiscretization().getTimeDiscretization().getTime(numberOfTimeSteps - (m + 1));

			final TwoStateStepResult next = performStableDouglasStep(uInactive, uActive, currentTime, dt);

			uInactive = next.uInactive;
			uActive = next.uActive;

			applyTwoStateConditions(currentTime, uInactive, uActive);

			uInactive = sanitize(uInactive);
			uActive = sanitize(uActive);

			solutionSurface.setColumn(m + 1, uInactive.clone());
		}

		return solutionSurface.getData();
	}

	public double[] getValue(
			final double evaluationTime,
			final double time,
			final DoubleUnaryOperator valueAtMaturity) {
		final RealMatrix values = new Array2DRowRealMatrix(getValues(time, valueAtMaturity));
		final double tau = time - evaluationTime;
		final int timeIndex = model.getSpaceTimeDiscretization().getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);
		return values.getColumn(timeIndex);
	}

	public double[] getValue(
			final double evaluationTime,
			final double time,
			final DoubleBinaryOperator valueAtMaturity) {
		final RealMatrix values = new Array2DRowRealMatrix(getValues(time, valueAtMaturity));
		final double tau = time - evaluationTime;
		final int timeIndex = model.getSpaceTimeDiscretization().getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);
		return values.getColumn(timeIndex);
	}

	private void initializeTerminalStates(
			final double[] uInactive,
			final double[] uActive,
			final DoubleBinaryOperator valueAtMaturity) {

		final BarrierType barrierType = product.getBarrierType();
		final double barrier = product.getBarrierValue();

		for(int j = 0; j < nV; j++) {
			for(int i = 0; i < nS; i++) {
				final int k = flatten(i, j);
				final double s = sGrid[i];
				final double v = vGrid[j];
				final double payoff = valueAtMaturity.applyAsDouble(s, v);

				uActive[k] = payoff;
				uInactive[k] = isAlreadyHitRegion(s, barrierType, barrier) ? payoff : product.getRebate();
			}
		}
	}

	/**
	 * Conservative stabilization-first step:
	 * split one PDE step into two half Douglas ADI steps for both states.
	 */
	private TwoStateStepResult performStableDouglasStep(
			final double[] uInactive,
			final double[] uActive,
			final double currentTime,
			final double dt) {

		final double halfDt = 0.5 * dt;

		TwoStateStepResult mid = performDouglasHalfStep(uInactive, uActive, currentTime + halfDt, halfDt);
		applyTwoStateConditions(currentTime + halfDt, mid.uInactive, mid.uActive);
		mid = new TwoStateStepResult(sanitize(mid.uInactive), sanitize(mid.uActive));

		TwoStateStepResult next = performDouglasHalfStep(mid.uInactive, mid.uActive, currentTime, halfDt);
		applyTwoStateConditions(currentTime, next.uInactive, next.uActive);

		return new TwoStateStepResult(sanitize(next.uInactive), sanitize(next.uActive));
	}

	private TwoStateStepResult performDouglasHalfStep(
			final double[] uInactive,
			final double[] uActive,
			final double currentTime,
			final double dt) {

		final double[] explicitInactive = applyFullExplicitOperator(uInactive, currentTime);
		final double[] explicitActive = applyFullExplicitOperator(uActive, currentTime);

		double[] y0Inactive = add(uInactive, scale(explicitInactive, dt));
		double[] y0Active = add(uActive, scale(explicitActive, dt));

		applyTwoStateConditions(currentTime, y0Inactive, y0Active);

		final double[] a1Inactive = applyA1Explicit(uInactive, currentTime);
		final double[] a1Active = applyA1Explicit(uActive, currentTime);

		final double[] rhs1Inactive = subtract(y0Inactive, scale(a1Inactive, theta * dt));
		final double[] rhs1Active = subtract(y0Active, scale(a1Active, theta * dt));

		double[] y1Inactive = solveSpotLinesInactive(rhs1Inactive, currentTime, dt);
		double[] y1Active = solveSpotLinesActive(rhs1Active, currentTime, dt);

		applyTwoStateConditions(currentTime, y1Inactive, y1Active);

		final double[] a2Inactive = applyA2Explicit(uInactive, currentTime);
		final double[] a2Active = applyA2Explicit(uActive, currentTime);

		final double[] rhs2Inactive = subtract(y1Inactive, scale(a2Inactive, theta * dt));
		final double[] rhs2Active = subtract(y1Active, scale(a2Active, theta * dt));

		double[] y2Inactive = solveVarianceLinesInactive(rhs2Inactive, currentTime, dt);
		double[] y2Active = solveVarianceLinesActive(rhs2Active, currentTime, dt);

		applyTwoStateConditions(currentTime, y2Inactive, y2Active);

		return new TwoStateStepResult(y2Inactive, y2Active);
	}

	private void applyTwoStateConditions(
			final double currentTime,
			final double[] uInactive,
			final double[] uActive) {

		applyActiveOuterBoundaries(currentTime, uActive);
		applyInactiveContinuationBoundary(currentTime, uInactive);
		applyCouplingOnHitSet(uInactive, uActive);
	}

	private void applyActiveOuterBoundaries(final double currentTime, final double[] uActive) {

		for(int j = 0; j < nV; j++) {
			final double v = vGrid[j];

			final double[] lowerValues = activeBoundaryProvider.getLowerBoundaryValues(currentTime, sGrid[0], v);
			final double[] upperValues = activeBoundaryProvider.getUpperBoundaryValues(currentTime, sGrid[nS - 1], v);

			if(!Double.isNaN(lowerValues[0])) {
				uActive[flatten(0, j)] = lowerValues[0];
			}
			if(!Double.isNaN(upperValues[0])) {
				uActive[flatten(nS - 1, j)] = upperValues[0];
			}
		}

		for(int i = 0; i < nS; i++) {
			final double s = sGrid[i];

			final double[] lowerValues = activeBoundaryProvider.getLowerBoundaryValues(currentTime, s, vGrid[0]);
			final double[] upperValues = activeBoundaryProvider.getUpperBoundaryValues(currentTime, s, vGrid[nV - 1]);

			if(!Double.isNaN(lowerValues[1])) {
				uActive[flatten(i, 0)] = lowerValues[1];
			}
			if(!Double.isNaN(upperValues[1])) {
				uActive[flatten(i, nV - 1)] = upperValues[1];
			}
		}
	}

	private void applyInactiveContinuationBoundary(final double currentTime, final double[] uInactive) {

		final double noHitValue = getDiscountedNoHitValue(currentTime);
		final BarrierType barrierType = product.getBarrierType();

		if(barrierType == BarrierType.DOWN_IN) {
			for(int j = 0; j < nV; j++) {
				uInactive[flatten(nS - 1, j)] = noHitValue;
			}
		}
		else if(barrierType == BarrierType.UP_IN) {
			for(int j = 0; j < nV; j++) {
				uInactive[flatten(0, j)] = noHitValue;
			}
		}
	}

	private void applyCouplingOnHitSet(final double[] uInactive, final double[] uActive) {

		final BarrierType barrierType = product.getBarrierType();
		final double barrier = product.getBarrierValue();

		for(int j = 0; j < nV; j++) {
			for(int i = 0; i < nS; i++) {
				final int k = flatten(i, j);
				if(isAlreadyHitRegion(sGrid[i], barrierType, barrier)) {
					uInactive[k] = uActive[k];
				}
			}
		}
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

		final double dfNow = model.getRiskFreeCurve().getDiscountFactor(t);
		final double dfMat = model.getRiskFreeCurve().getDiscountFactor(maturity);

		return product.getRebate() * dfMat / dfNow;
	}

	protected double[] applyFullExplicitOperator(final double[] u, final double time) {
		return add(add(applyA0Explicit(u, time), applyA1Explicit(u, time)), applyA2Explicit(u, time));
	}

	/**
	 * Explicit mixed-derivative plus discount operator.
	 */
	protected double[] applyA0Explicit(final double[] u, final double time) {

		final double[] out = new double[n];

		final double tSafe = Math.max(time, 1E-10);
		final double discountFactor = model.getRiskFreeCurve().getDiscountFactor(tSafe);
		final double r = -Math.log(discountFactor) / tSafe;

		for(int j = 1; j < nV - 1; j++) {
			for(int i = 1; i < nS - 1; i++) {
				final int k = flatten(i, j);

				final double s = sGrid[i];
				final double v = vGrid[j];

				final double[][] b = model.getFactorLoading(time, s, v);

				double aSV = 0.0;
				for(int f = 0; f < b[0].length; f++) {
					aSV += b[0][f] * b[1][f];
				}

				final double dsDown = sGrid[i] - sGrid[i - 1];
				final double dsUp = sGrid[i + 1] - sGrid[i];
				final double dvDown = vGrid[j] - vGrid[j - 1];
				final double dvUp = vGrid[j + 1] - vGrid[j];

				final double dSdV =
						(
								u[flatten(i + 1, j + 1)]
								- u[flatten(i + 1, j - 1)]
								- u[flatten(i - 1, j + 1)]
								+ u[flatten(i - 1, j - 1)]
						)
						/ ((dsDown + dsUp) * (dvDown + dvUp));

				out[k] = aSV * dSdV - r * u[k];
			}
		}

		return out;
	}

	/**
	 * Explicit spot-direction operator.
	 */
	protected double[] applyA1Explicit(final double[] u, final double time) {

		final double[] out = new double[n];

		for(int j = 0; j < nV; j++) {
			for(int i = 1; i < nS - 1; i++) {
				final int k = flatten(i, j);

				final double s = sGrid[i];
				final double v = vGrid[j];

				final double[] drift = model.getDrift(time, s, v);
				final double[][] b = model.getFactorLoading(time, s, v);

				final double muS = drift[0];

				double aSS = 0.0;
				for(int f = 0; f < b[0].length; f++) {
					aSS += b[0][f] * b[0][f];
				}

				final double dsDown = sGrid[i] - sGrid[i - 1];
				final double dsUp = sGrid[i + 1] - sGrid[i];

				final double dS =
						(u[flatten(i + 1, j)] - u[flatten(i - 1, j)])
						/ (dsDown + dsUp);

				final double dSS =
						2.0 * (
								(u[flatten(i + 1, j)] - u[k]) / dsUp
								- (u[k] - u[flatten(i - 1, j)]) / dsDown
						)
						/ (dsDown + dsUp);

				out[k] = muS * dS + 0.5 * aSS * dSS;
			}
		}

		return out;
	}

	/**
	 * Explicit variance-direction operator.
	 */
	protected double[] applyA2Explicit(final double[] u, final double time) {

		final double[] out = new double[n];

		for(int j = 1; j < nV - 1; j++) {
			for(int i = 0; i < nS; i++) {
				final int k = flatten(i, j);

				final double s = sGrid[i];
				final double v = vGrid[j];

				final double[] drift = model.getDrift(time, s, v);
				final double[][] b = model.getFactorLoading(time, s, v);

				final double muV = drift[1];

				double aVV = 0.0;
				for(int f = 0; f < b[1].length; f++) {
					aVV += b[1][f] * b[1][f];
				}

				final double dvDown = vGrid[j] - vGrid[j - 1];
				final double dvUp = vGrid[j + 1] - vGrid[j];

				final double dV =
						(u[flatten(i, j + 1)] - u[flatten(i, j - 1)])
						/ (dvDown + dvUp);

				final double dVV =
						2.0 * (
								(u[flatten(i, j + 1)] - u[k]) / dvUp
								- (u[k] - u[flatten(i, j - 1)]) / dvDown
						)
						/ (dvDown + dvUp);

				out[k] = muV * dV + 0.5 * aVV * dVV;
			}
		}

		return out;
	}

	/**
	 * Implicit solve along spot lines for fixed variance in the active regime.
	 * Uses the same stabilized line solve style as the vanilla solver, with active
	 * spot boundaries injected from the provider.
	 */
	protected double[] solveSpotLinesActive(
			final double[] rhs,
			final double time,
			final double dt) {

		final double[] out = rhs.clone();

		for(int j = 0; j < nV; j++) {
			final TridiagonalMatrix m = stencilBuilder.buildSpotLineMatrix(time, dt, theta, j);

			final double[] lineRhs = new double[nS];
			for(int i = 0; i < nS; i++) {
				lineRhs[i] = rhs[flatten(i, j)];
			}

			final double v = vGrid[j];
			final double[] lowerValues = activeBoundaryProvider.getLowerBoundaryValues(time, sGrid[0], v);
			final double[] upperValues = activeBoundaryProvider.getUpperBoundaryValues(time, sGrid[nS - 1], v);

			final double lowerBoundaryValue = Double.isNaN(lowerValues[0]) ? lineRhs[0] : lowerValues[0];
			final double upperBoundaryValue = Double.isNaN(upperValues[0]) ? lineRhs[nS - 1] : upperValues[0];

			overwriteBoundaryRow(m, lineRhs, 0, lowerBoundaryValue);
			overwriteBoundaryRow(m, lineRhs, nS - 1, upperBoundaryValue);

			final double[] solved = ThomasSolver.solve(m.lower, m.diag, m.upper, lineRhs);

			for(int i = 0; i < nS; i++) {
				out[flatten(i, j)] = solved[i];
			}
		}

		return out;
	}

	/**
	 * Implicit solve along spot lines for fixed variance in the inactive regime.
	 * Uses continuation-side no-hit boundary and leaves the hit-side boundary to the
	 * two-state overlay re-imposition.
	 */
	protected double[] solveSpotLinesInactive(
			final double[] rhs,
			final double time,
			final double dt) {

		final double[] out = rhs.clone();
		final double noHitValue = getDiscountedNoHitValue(time);
		final BarrierType barrierType = product.getBarrierType();

		for(int j = 0; j < nV; j++) {
			final TridiagonalMatrix m = stencilBuilder.buildSpotLineMatrix(time, dt, theta, j);

			final double[] lineRhs = new double[nS];
			for(int i = 0; i < nS; i++) {
				lineRhs[i] = rhs[flatten(i, j)];
			}

			final double lowerBoundaryValue;
			final double upperBoundaryValue;

			if(barrierType == BarrierType.DOWN_IN) {
				lowerBoundaryValue = lineRhs[0];
				upperBoundaryValue = noHitValue;
			}
			else if(barrierType == BarrierType.UP_IN) {
				lowerBoundaryValue = noHitValue;
				upperBoundaryValue = lineRhs[nS - 1];
			}
			else {
				throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
			}

			overwriteBoundaryRow(m, lineRhs, 0, lowerBoundaryValue);
			overwriteBoundaryRow(m, lineRhs, nS - 1, upperBoundaryValue);

			final double[] solved = ThomasSolver.solve(m.lower, m.diag, m.upper, lineRhs);

			for(int i = 0; i < nS; i++) {
				out[flatten(i, j)] = solved[i];
			}
		}

		return out;
	}

	/**
	 * Implicit solve along variance lines for fixed spot in the active regime.
	 * Variance boundaries are taken from the provider where supplied, otherwise from rhs.
	 */
	protected double[] solveVarianceLinesActive(
			final double[] rhs,
			final double time,
			final double dt) {

		final double[] out = rhs.clone();

		for(int i = 0; i < nS; i++) {
			final TridiagonalMatrix m = stencilBuilder.buildVarianceLineMatrix(time, dt, theta, i);

			final double[] lineRhs = new double[nV];
			for(int j = 0; j < nV; j++) {
				lineRhs[j] = rhs[flatten(i, j)];
			}

			final double s = sGrid[i];
			final double[] lowerValues = activeBoundaryProvider.getLowerBoundaryValues(time, s, vGrid[0]);
			final double[] upperValues = activeBoundaryProvider.getUpperBoundaryValues(time, s, vGrid[nV - 1]);

			final double lowerBoundaryValue = Double.isNaN(lowerValues[1]) ? lineRhs[0] : lowerValues[1];
			final double upperBoundaryValue = Double.isNaN(upperValues[1]) ? lineRhs[nV - 1] : upperValues[1];

			overwriteBoundaryRow(m, lineRhs, 0, lowerBoundaryValue);
			overwriteBoundaryRow(m, lineRhs, nV - 1, upperBoundaryValue);

			final double[] solved = ThomasSolver.solve(m.lower, m.diag, m.upper, lineRhs);

			for(int j = 0; j < nV; j++) {
				out[flatten(i, j)] = solved[j];
			}
		}

		return out;
	}

	/**
	 * Implicit solve along variance lines for fixed spot in the inactive regime.
	 * Keeps incoming variance-edge values, like the stabilized vanilla solver.
	 */
	protected double[] solveVarianceLinesInactive(
			final double[] rhs,
			final double time,
			final double dt) {

		final double[] out = rhs.clone();

		for(int i = 0; i < nS; i++) {
			final TridiagonalMatrix m = stencilBuilder.buildVarianceLineMatrix(time, dt, theta, i);

			final double[] lineRhs = new double[nV];
			for(int j = 0; j < nV; j++) {
				lineRhs[j] = rhs[flatten(i, j)];
			}

			overwriteBoundaryRow(m, lineRhs, 0, lineRhs[0]);
			overwriteBoundaryRow(m, lineRhs, nV - 1, lineRhs[nV - 1]);

			final double[] solved = ThomasSolver.solve(m.lower, m.diag, m.upper, lineRhs);

			for(int j = 0; j < nV; j++) {
				out[flatten(i, j)] = solved[j];
			}
		}

		return out;
	}

	private void overwriteBoundaryRow(
			final TridiagonalMatrix m,
			final double[] rhs,
			final int row,
			final double value) {

		m.lower[row] = 0.0;
		m.diag[row] = 1.0;
		m.upper[row] = 0.0;
		rhs[row] = value;
	}

	private void validateBarrierOnGridNode(final double[] grid, final double barrier) {
		for(final double x : grid) {
			if(Math.abs(x - barrier) < 1E-12) {
				return;
			}
		}
		throw new IllegalArgumentException(
				"Barrier must coincide with a first-state-variable grid node for direct two-state Heston pricing.");
	}

	private boolean isAlreadyHitRegion(
			final double s,
			final BarrierType barrierType,
			final double barrier) {

		switch(barrierType) {
		case DOWN_IN:
			return s <= barrier;
		case UP_IN:
			return s >= barrier;
		default:
			throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
		}
	}

	private int flatten(final int iS, final int iV) {
		return iS + iV * nS;
	}

	private double[] add(final double[] a, final double[] b) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = a[i] + b[i];
		}
		return out;
	}

	private double[] subtract(final double[] a, final double[] b) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = a[i] - b[i];
		}
		return out;
	}

	private double[] scale(final double[] a, final double c) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = c * a[i];
		}
		return out;
	}

	private double[] sanitize(final double[] u) {
		final double[] out = new double[u.length];
		for(int i = 0; i < u.length; i++) {
			final double value = u[i];
			if(!Double.isFinite(value)) {
				out[i] = 0.0;
			}
			else if(value > 1E12) {
				out[i] = 1E12;
			}
			else if(value < -1E12) {
				out[i] = -1E12;
			}
			else {
				out[i] = value;
			}
		}
		return out;
	}

	private static final class TwoStateStepResult {
		private final double[] uInactive;
		private final double[] uActive;

		private TwoStateStepResult(final double[] uInactive, final double[] uActive) {
			this.uInactive = uInactive;
			this.uActive = uActive;
		}
	}
}