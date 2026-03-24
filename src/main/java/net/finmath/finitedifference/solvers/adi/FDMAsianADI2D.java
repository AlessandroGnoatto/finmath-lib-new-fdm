package net.finmath.finitedifference.solvers.adi;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.solvers.ThomasSolver;
import net.finmath.finitedifference.solvers.TridiagonalMatrix;
import net.finmath.modelling.Exercise;

/**
 * Specialized ADI solver for arithmetic Asian options in lifted state (S, I),
 * where
 *
 *   dS_t = mu_S dt + sigma(S,t) dW_t,
 *   dI_t = S_t dt.
 *
 * In time-to-maturity tau = T - t, the backward pricing PDE is
 *
 *   u_tau = A0 u + A1 u + A2 u
 *
 * with
 *
 *   A0 u = -r u
 *   A1 u = mu_S u_S + 0.5 a_SS u_SS
 *   A2 u = S u_I
 *
 * There is:
 * - no diffusion in the I direction
 * - no mixed derivative
 *
 * The important point is that the I direction is pure transport. In tau-time,
 * the correct upwind direction is FORWARD in I.
 */
public class FDMAsianADI2D extends AbstractADI2D {

	public FDMAsianADI2D(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise) {
		super(model, product, spaceTimeDiscretization, exercise);
	}

	/**
	 * For the lifted Asian PDE, A0 contains only discounting.
	 * No mixed derivative term is present.
	 */
	@Override
	protected double[] applyA0Explicit(final double[] u, final double time) {
		final double[] out = new double[n];

		final double tSafe = Math.max(time, 1E-10);
		final double discountFactor = model.getRiskFreeCurve().getDiscountFactor(tSafe);
		final double r = -Math.log(discountFactor) / tSafe;

		for(int j = 0; j < n1; j++) {
			for(int i = 0; i < n0; i++) {
				out[flatten(i, j)] = -r * u[flatten(i, j)];
			}
		}

		return out;
	}

	/**
	 * Explicit application of A2 u = S * u_I.
	 *
	 * IMPORTANT:
	 * In time-to-maturity tau, the transport is upwinded FORWARD in I.
	 *
	 * For interior nodes:
	 *
	 *   u_I(S_i, I_j) ~ (u_{i,j+1} - u_{i,j}) / (I_{j+1} - I_j)
	 *
	 * The top row j = n1 - 1 is handled via boundary overwrite later.
	 */
	@Override
	protected double[] applyA2Explicit(final double[] u, final double time) {
		final double[] out = new double[n];

		for(int i = 0; i < n0; i++) {
			final double s = x0Grid[i];

			for(int j = 0; j < n1 - 1; j++) {
				final double dIUp = x1Grid[j + 1] - x1Grid[j];
				out[flatten(i, j)] = s * (u[flatten(i, j + 1)] - u[flatten(i, j)]) / dIUp;
			}

			/*
			 * Top I boundary handled separately by Dirichlet overwrite.
			 */
			out[flatten(i, n1 - 1)] = 0.0;
		}

		return out;
	}

	/**
	 * Implicit line solves in the I direction for
	 *
	 *   (I - theta dt A2) v = rhs
	 *
	 * with A2 u = S u_I and forward upwinding:
	 *
	 *   u_I(S_i, I_j) ~ (u_{i,j+1} - u_{i,j}) / (I_{j+1} - I_j)
	 *
	 * Hence for j = 0,...,n1-2:
	 *
	 *   v_j - theta dt S_i (v_{j+1} - v_j)/dIUp = rhs_j
	 *
	 * i.e.
	 *
	 *   (1 + lambda) v_j - lambda v_{j+1} = rhs_j,
	 *   lambda = theta dt S_i / dIUp
	 *
	 * So the line matrix is upper bidiagonal in the interior.
	 *
	 * Boundary policy:
	 * - I = 0      : natural / outflow side -> identity row
	 * - I = I_max  : Dirichlet if prescribed by product/model
	 */
	@Override
	protected double[] solveSecondDirectionLines(
			final double[] rhs,
			final double time,
			final double dt) {

		final double[] out = rhs.clone();

		for(int i = 0; i < n0; i++) {
			final double s = x0Grid[i];

			final TridiagonalMatrix m = new TridiagonalMatrix(n1);
			final double[] lineRhs = new double[n1];

			for(int j = 0; j < n1; j++) {
				lineRhs[j] = rhs[flatten(i, j)];
			}

			/*
			 * Lower I boundary (j = 0): natural / no Dirichlet by default.
			 * We use an identity row unless overwritten by the product/model.
			 */
			m.lower[0] = 0.0;
			m.diag[0] = 1.0;
			m.upper[0] = 0.0;

			/*
			 * Interior transport rows: j = 1,...,n1-2
			 * We also set j = 0 in transport form below, but immediately replace it
			 * by the identity row above, which is clearer and safer.
			 */
			for(int j = 0; j < n1 - 1; j++) {
				final double dIUp = x1Grid[j + 1] - x1Grid[j];
				final double lambda = theta * dt * s / dIUp;

				m.lower[j] = 0.0;
				m.diag[j] = 1.0 + lambda;
				m.upper[j] = -lambda;
			}

			/*
			 * Restore lower boundary row explicitly.
			 */
			m.lower[0] = 0.0;
			m.diag[0] = 1.0;
			m.upper[0] = 0.0;

			/*
			 * Upper I boundary: use product/model boundary if Dirichlet.
			 * This is the inflow side for the transport in tau-time.
			 */
			final double upperBoundaryValue =
					getUpperBoundaryValueForSecondDirection(time, i, lineRhs[n1 - 1]);
			overwriteBoundaryRow(m, lineRhs, n1 - 1, upperBoundaryValue);

			/*
			 * If the lower I boundary is ever defined as Dirichlet by the product/model,
			 * honor it as well.
			 */
			final double lowerBoundaryValue =
					getLowerBoundaryValueForSecondDirection(time, i, lineRhs[0]);
			overwriteBoundaryRow(m, lineRhs, 0, lowerBoundaryValue);

			final double[] solved = ThomasSolver.solve(m.lower, m.diag, m.upper, lineRhs);

			for(int j = 0; j < n1; j++) {
				out[flatten(i, j)] = solved[j];
			}
		}

		return out;
	}
}