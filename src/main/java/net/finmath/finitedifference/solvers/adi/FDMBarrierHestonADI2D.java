package net.finmath.finitedifference.solvers.adi;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.BarrierOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.solvers.ThomasSolver;
import net.finmath.finitedifference.solvers.TridiagonalMatrix;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.CallOrPut;

/**
 * Specialized 2D ADI solver for barrier options under a Heston state (S, v).
 *
 * <p>
 * The solver supports two modes:
 * </p>
 * <ul>
 *   <li>OUT: direct knock-out pricing on the alive region,</li>
 *   <li>IN_PRE_HIT: direct pre-hit knock-in pricing on the not-yet-hit region.</li>
 * </ul>
 *
 * <p>
 * In {@code IN_PRE_HIT} mode, an optional {@link BarrierPreHitSpecification}
 * may be supplied. If present, it defines the barrier-side Dirichlet trace
 * in the first spatial direction.
 * </p>
 *
 * <p>
 * Heston-specific treatment:
 * </p>
 * <ul>
 *   <li>on the degenerate variance boundary {@code v = 0}, the first spatial
 *       line uses a drift-only upwind discretization,</li>
 *   <li>in {@code IN_PRE_HIT} mode, the barrier node is pinned to the
 *       activated trace after each directional substep, with no additional
 *       overwrite of the first interior node.</li>
 * </ul>
 */
public class FDMBarrierHestonADI2D extends AbstractADI2D {

	private final ADI2DStencilBuilder stencilBuilder;
	private final BarrierPDEMode mode;
	private final BarrierPreHitSpecification preHitSpecification;

	private static final double SMALL_TIME = 1E-12;

	public FDMBarrierHestonADI2D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final BarrierPDEMode mode) {
		this(model, product, spaceTimeDiscretization, exercise, mode, null);
	}

	public FDMBarrierHestonADI2D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final BarrierPDEMode mode,
			final BarrierPreHitSpecification preHitSpecification) {
		super(model, product, spaceTimeDiscretization, exercise);
		this.stencilBuilder = new ADI2DStencilBuilder(model, x0Grid, x1Grid);
		this.mode = mode;
		this.preHitSpecification = preHitSpecification;
	}

	public FDMBarrierHestonADI2D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final BarrierPDEMode mode,
			final AbstractADI2D.ADIScheme adiScheme) {
		this(model, product, spaceTimeDiscretization, exercise, mode, null, adiScheme);
	}

	public FDMBarrierHestonADI2D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final BarrierPDEMode mode,
			final BarrierPreHitSpecification preHitSpecification,
			final AbstractADI2D.ADIScheme adiScheme) {
		super(model, product, spaceTimeDiscretization, exercise, adiScheme);
		this.stencilBuilder = new ADI2DStencilBuilder(model, x0Grid, x1Grid);
		this.mode = mode;
		this.preHitSpecification = preHitSpecification;
	}

	public BarrierPDEMode getMode() {
		return mode;
	}

	public BarrierPreHitSpecification getPreHitSpecification() {
		return preHitSpecification;
	}

	@Override
	protected double[] solveFirstDirectionLines(
			final double[] rhs,
			final double time,
			final double dt) {

		final double[] out = rhs.clone();
		enforceBarrierTraceIfNeeded(out, time);

		for(int i1 = 0; i1 < n1; i1++) {

			final TridiagonalMatrix m;
			if(i1 == 0) {
				m = buildDegenerateVarianceFirstDirectionLineMatrix(time, dt);
			}
			else {
				m = stencilBuilder.buildFirstDirectionLineMatrix(time, dt, theta, i1);
			}

			final double[] lineRhs = new double[n0];
			for(int i0 = 0; i0 < n0; i0++) {
				lineRhs[i0] = out[flatten(i0, i1)];
			}

			if(usesActivatedBarrierTrace()) {
				if(preHitSpecification.isDownIn()) {
					overwriteBoundaryRow(m, lineRhs, 0, getActivatedTraceValue(i1, time));
				}
				else if(preHitSpecification.isUpIn()) {
					overwriteBoundaryRow(m, lineRhs, n0 - 1, getActivatedTraceValue(i1, time));
				}
				else {
					throw new IllegalStateException("Unsupported pre-hit barrier type.");
				}
			}
			else {
				final BoundaryCondition[] lowerConditions =
						model.getBoundaryConditionsAtLowerBoundary(product, time, x0Grid[0], x1Grid[i1]);

				if(lowerConditions != null
						&& lowerConditions.length > 0
						&& lowerConditions[0] != null
						&& lowerConditions[0].isDirichlet()) {
					overwriteBoundaryRow(m, lineRhs, 0, lowerConditions[0].getValue());
				}

				final BoundaryCondition[] upperConditions =
						model.getBoundaryConditionsAtUpperBoundary(product, time, x0Grid[n0 - 1], x1Grid[i1]);

				if(upperConditions != null
						&& upperConditions.length > 0
						&& upperConditions[0] != null
						&& upperConditions[0].isDirichlet()) {
					overwriteBoundaryRow(m, lineRhs, n0 - 1, upperConditions[0].getValue());
				}
			}

			final double[] solved = ThomasSolver.solve(m.lower, m.diag, m.upper, lineRhs);

			for(int i0 = 0; i0 < n0; i0++) {
				out[flatten(i0, i1)] = solved[i0];
			}
		}

		enforceBarrierTraceIfNeeded(out, time);
		return out;
	}

	@Override
	protected double[] solveSecondDirectionLines(
			final double[] rhs,
			final double time,
			final double dt) {

		final double[] out = rhs.clone();
		enforceBarrierTraceIfNeeded(out, time);

		for(int i0 = 0; i0 < n0; i0++) {
			final TridiagonalMatrix m = stencilBuilder.buildSecondDirectionLineMatrix(time, dt, theta, i0);

			final double[] lineRhs = new double[n1];
			for(int i1 = 0; i1 < n1; i1++) {
				lineRhs[i1] = out[flatten(i0, i1)];
			}

			final BoundaryCondition[] lowerConditions =
					model.getBoundaryConditionsAtLowerBoundary(product, time, x0Grid[i0], x1Grid[0]);

			if(lowerConditions != null
					&& lowerConditions.length > 1
					&& lowerConditions[1] != null
					&& lowerConditions[1].isDirichlet()) {
				overwriteBoundaryRow(m, lineRhs, 0, lowerConditions[1].getValue());
			}

			final BoundaryCondition[] upperConditions =
					model.getBoundaryConditionsAtUpperBoundary(product, time, x0Grid[i0], x1Grid[n1 - 1]);

			if(upperConditions != null
					&& upperConditions.length > 1
					&& upperConditions[1] != null
					&& upperConditions[1].isDirichlet()) {
				overwriteBoundaryRow(m, lineRhs, n1 - 1, upperConditions[1].getValue());
			}

			final double[] solved = ThomasSolver.solve(m.lower, m.diag, m.upper, lineRhs);

			for(int i1 = 0; i1 < n1; i1++) {
				out[flatten(i0, i1)] = solved[i1];
			}
		}

		enforceBarrierTraceIfNeeded(out, time);
		return out;
	}

	private TridiagonalMatrix buildDegenerateVarianceFirstDirectionLineMatrix(
			final double time,
			final double dt) {

		final TridiagonalMatrix m = new TridiagonalMatrix(n0);
		final double tSafe = Math.max(time, SMALL_TIME);

		for(int i0 = 1; i0 < n0 - 1; i0++) {
			final double s = x0Grid[i0];
			final double[] drift = model.getDrift(tSafe, s, x1Grid[0]);
			final double muS = drift[0];

			final double dxDown = x0Grid[i0] - x0Grid[i0 - 1];
			final double dxUp = x0Grid[i0 + 1] - x0Grid[i0];

			if(muS >= 0.0) {
				final double lambda = theta * dt * muS / dxDown;
				m.lower[i0] = -lambda;
				m.diag[i0] = 1.0 + lambda;
				m.upper[i0] = 0.0;
			}
			else {
				final double lambda = theta * dt * muS / dxUp;
				m.lower[i0] = 0.0;
				m.diag[i0] = 1.0 - lambda;
				m.upper[i0] = lambda;
			}
		}

		m.lower[0] = 0.0;
		m.diag[0] = 1.0;
		m.upper[0] = 0.0;

		m.lower[n0 - 1] = 0.0;
		m.diag[n0 - 1] = 1.0;
		m.upper[n0 - 1] = 0.0;

		return m;
	}

	private boolean usesActivatedBarrierTrace() {
		return mode == BarrierPDEMode.IN_PRE_HIT && preHitSpecification != null;
	}

	private double getActivatedTraceValue(final int secondStateIndex, final double time) {
		final ActivatedBarrierTrace2D trace = preHitSpecification.getActivatedBarrierTrace();

		if(secondStateIndex < 0 || secondStateIndex >= trace.getNumberOfSecondStatePoints()) {
			throw new IllegalArgumentException("secondStateIndex out of range.");
		}

		final int timeIndex = spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(time);
		final int boundedTimeIndex = Math.max(0, Math.min(timeIndex, trace.getNumberOfTimePoints() - 1));

		return trace.getValue(secondStateIndex, boundedTimeIndex);
	}

	private void valuesAtBarrier(final double[] values, final double time, final int barrierIndex) {
		for(int i1 = 0; i1 < n1; i1++) {
			values[flatten(barrierIndex, i1)] = getActivatedTraceValue(i1, time);
		}
	}

	private void enforceBarrierTraceIfNeeded(final double[] values, final double time) {
		if(!usesActivatedBarrierTrace()) {
			return;
		}

		final int barrierIndex;
		if(preHitSpecification.isDownIn()) {
			barrierIndex = 0;
		}
		else if(preHitSpecification.isUpIn()) {
			barrierIndex = n0 - 1;
		}
		else {
			throw new IllegalStateException("Unsupported pre-hit barrier type.");
		}

		valuesAtBarrier(values, time, barrierIndex);
	}
}