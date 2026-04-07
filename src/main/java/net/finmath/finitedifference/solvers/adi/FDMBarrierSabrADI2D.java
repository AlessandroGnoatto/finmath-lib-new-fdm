package net.finmath.finitedifference.solvers.adi;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.solvers.ThomasSolver;
import net.finmath.finitedifference.solvers.TridiagonalMatrix;
import net.finmath.modelling.Exercise;

/**
 * Specialized 2D ADI solver for barrier options under a SABR state (S, alpha).
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
 * In both cases, boundary rows are overwritten only when the model boundary
 * condition is explicitly Dirichlet. If the boundary is {@code none()}, the PDE
 * row is left intact.
 * </p>
 *
 * <p>
 * In {@code IN_PRE_HIT} mode, an optional {@link BarrierPreHitSpecification}
 * may be supplied. If present, it defines the barrier-side Dirichlet trace
 * in the first spatial direction.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class FDMBarrierSabrADI2D extends AbstractADI2D {

	private final BarrierPDEMode mode;
	private final BarrierPreHitSpecification preHitSpecification;

	public FDMBarrierSabrADI2D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final BarrierPDEMode mode) {
		this(model, product, spaceTimeDiscretization, exercise, mode, null);
	}

	public FDMBarrierSabrADI2D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final BarrierPDEMode mode,
			final BarrierPreHitSpecification preHitSpecification) {
		super(model, product, spaceTimeDiscretization, exercise);
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

		for(int j = 0; j < n1; j++) {
			final TridiagonalMatrix m = stencilBuilder.buildFirstDirectionLineMatrix(time, dt, theta, j);

			final double[] lineRhs = new double[n0];
			for(int i = 0; i < n0; i++) {
				lineRhs[i] = rhs[flatten(i, j)];
			}

			if(usesActivatedBarrierTrace()) {
				if(preHitSpecification.isDownIn()) {
					overwriteBoundaryRow(
							m,
							lineRhs,
							0,
							getActivatedTraceValue(j, time)
					);
				}
				else if(preHitSpecification.isUpIn()) {
					overwriteBoundaryRow(
							m,
							lineRhs,
							n0 - 1,
							getActivatedTraceValue(j, time)
					);
				}
			}
			else {
				final BoundaryCondition[] lowerConditions =
						model.getBoundaryConditionsAtLowerBoundary(product, time, x0Grid[0], x1Grid[j]);

				if(lowerConditions != null
						&& lowerConditions.length > 0
						&& lowerConditions[0] != null
						&& lowerConditions[0].isDirichlet()) {
					overwriteBoundaryRow(m, lineRhs, 0, lowerConditions[0].getValue());
				}

				final BoundaryCondition[] upperConditions =
						model.getBoundaryConditionsAtUpperBoundary(product, time, x0Grid[n0 - 1], x1Grid[j]);

				if(upperConditions != null
						&& upperConditions.length > 0
						&& upperConditions[0] != null
						&& upperConditions[0].isDirichlet()) {
					overwriteBoundaryRow(m, lineRhs, n0 - 1, upperConditions[0].getValue());
				}
			}

			final double[] solved = ThomasSolver.solve(m.lower, m.diag, m.upper, lineRhs);

			for(int i = 0; i < n0; i++) {
				out[flatten(i, j)] = solved[i];
			}
		}

		return out;
	}

	@Override
	protected double[] solveSecondDirectionLines(
			final double[] rhs,
			final double time,
			final double dt) {

		final double[] out = rhs.clone();

		for(int i = 0; i < n0; i++) {
			final TridiagonalMatrix m = stencilBuilder.buildSecondDirectionLineMatrix(time, dt, theta, i);

			final double[] lineRhs = new double[n1];
			for(int j = 0; j < n1; j++) {
				lineRhs[j] = rhs[flatten(i, j)];
			}

			final BoundaryCondition[] lowerConditions =
					model.getBoundaryConditionsAtLowerBoundary(product, time, x0Grid[i], x1Grid[0]);

			if(lowerConditions != null
					&& lowerConditions.length > 1
					&& lowerConditions[1] != null
					&& lowerConditions[1].isDirichlet()) {
				overwriteBoundaryRow(m, lineRhs, 0, lowerConditions[1].getValue());
			}

			final BoundaryCondition[] upperConditions =
					model.getBoundaryConditionsAtUpperBoundary(product, time, x0Grid[i], x1Grid[n1 - 1]);

			if(upperConditions != null
					&& upperConditions.length > 1
					&& upperConditions[1] != null
					&& upperConditions[1].isDirichlet()) {
				overwriteBoundaryRow(m, lineRhs, n1 - 1, upperConditions[1].getValue());
			}

			final double[] solved = ThomasSolver.solve(m.lower, m.diag, m.upper, lineRhs);

			for(int j = 0; j < n1; j++) {
				out[flatten(i, j)] = solved[j];
			}
		}

		return out;
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
}