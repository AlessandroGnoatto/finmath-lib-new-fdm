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
 * This solver preserves the generic Douglas-type ADI splitting from {@link AbstractADI2D},
 * but respects {@link net.finmath.finitedifference.boundaries.StandardBoundaryCondition#none()}
 * in both spatial directions.
 * </p>
 *
 * <p>
 * This is required for direct pricing of SABR barriers, because the boundary implementation
 * may intentionally leave continuation-side spot boundaries and volatility boundaries free
 * via {@code none()}. The generic {@link AbstractADI2D} line solves overwrite boundary
 * rows unconditionally, which destroys that semantics.
 * </p>
 *
 * <p>
 * In particular:
 * </p>
 * <ul>
 *   <li>if a boundary is Dirichlet, we overwrite the row,</li>
 *   <li>if a boundary is {@code none()}, we keep the PDE row intact.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class FDMBarrierSabrADI2D extends AbstractADI2D {

	public FDMBarrierSabrADI2D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise) {
		super(model, product, spaceTimeDiscretization, exercise);
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

			/*
			 * S-direction lower boundary:
			 * overwrite only if explicitly Dirichlet.
			 */
			final BoundaryCondition[] lowerConditions =
					model.getBoundaryConditionsAtLowerBoundary(product, time, x0Grid[0], x1Grid[j]);

			if(lowerConditions != null
					&& lowerConditions.length > 0
					&& lowerConditions[0] != null
					&& lowerConditions[0].isDirichlet()) {
				overwriteBoundaryRow(m, lineRhs, 0, lowerConditions[0].getValue());
			}

			/*
			 * S-direction upper boundary:
			 * overwrite only if explicitly Dirichlet.
			 */
			final BoundaryCondition[] upperConditions =
					model.getBoundaryConditionsAtUpperBoundary(product, time, x0Grid[n0 - 1], x1Grid[j]);

			if(upperConditions != null
					&& upperConditions.length > 0
					&& upperConditions[0] != null
					&& upperConditions[0].isDirichlet()) {
				overwriteBoundaryRow(m, lineRhs, n0 - 1, upperConditions[0].getValue());
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

			/*
			 * alpha-direction lower boundary:
			 * overwrite only if explicitly Dirichlet.
			 */
			final BoundaryCondition[] lowerConditions =
					model.getBoundaryConditionsAtLowerBoundary(product, time, x0Grid[i], x1Grid[0]);

			if(lowerConditions != null
					&& lowerConditions.length > 1
					&& lowerConditions[1] != null
					&& lowerConditions[1].isDirichlet()) {
				overwriteBoundaryRow(m, lineRhs, 0, lowerConditions[1].getValue());
			}

			/*
			 * alpha-direction upper boundary:
			 * overwrite only if explicitly Dirichlet.
			 */
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
}