package net.finmath.finitedifference.solvers;

import java.util.function.DoubleUnaryOperator;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.ExerciseType;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;

/**
 * Theta-method solver for a one-dimensional PDE in <em>state-variable form</em>.
 *
 * <p>
 * This solver assumes the grid variable
 * {@code X} follows an SDE of the form
 * </p>
 *
 * <p>
 * {@code dX_t = mu(t, X_t) dt + sum_k b_k(t, X_t) dW_t^k}
 * </p>
 *
 * <p>
 * and constructs the backward PDE operator using
 * </p>
 *
 * <ul>
 *   <li>Drift term: {@code mu(t, x) * d/dx}</li>
 *   <li>Diffusion term: {@code 0.5 * a(t, x) * d^2/dx^2} where {@code a = sum_k b_k^2}</li>
 *   <li>Discounting term: {@code -r(t) * u}</li>
 * </ul>
 *
 * <p>
 * This makes the solver agnostic to whether {@code X} is {@code S}, {@code log S}, or any other monotone
 * transformation, as long as the model provides consistent coefficients for that chosen state variable.
 * </p>
 *
 * <p>
 * Boundary conditions are enforced as Dirichlet conditions by overwriting the first and last rows of the system using
 * {@link FiniteDifferenceEquityModel#getValueAtLowerBoundary(FiniteDifferenceProduct, double, double...)} and
 * {@link FiniteDifferenceEquityModel#getValueAtUpperBoundary(FiniteDifferenceProduct, double, double...)}. These are
 * interpreted as boundary values in the <em>state variable</em> coordinate.
 * </p>
 *
 * <p>
 * The returned matrix has dimension {@code nX x (nT+1)} and contains the full time history in time-to-maturity
 * coordinates: column 0 corresponds to {@code tau = 0} (maturity), column {@code m} corresponds to
 * {@code tau = timeDiscretization.getTime(m)}.
 * </p>
 *
 * @author Alessandro Gnoatto
 * @author Ralph Rudd
 * @author Christian Fries
 * @author Jörg Kienitz
 */
public class FDMThetaMethod1DStateVariableForm implements FDMSolver {

	private final FiniteDifferenceEquityModel model;
	private final FiniteDifferenceProduct product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final ExerciseType exercise;

	/**
	 * Creates a theta-method solver for a one-dimensional PDE in state-variable form.
	 *
	 * @param model The finite difference equity model providing drift, factor loadings, and boundary conditions.
	 * @param product The product used for boundary value queries.
	 * @param spaceTimeDiscretization The space-time discretization.
	 * @param exercise The exercise type (European/American).
	 */
	public FDMThetaMethod1DStateVariableForm(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final ExerciseType exercise) {
		this.model = model;
		this.product = product;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
		this.exercise = exercise;
	}

	@Override
	public double[][] getValues(final double time, final DoubleUnaryOperator valueAtMaturity) {

		// Full grid including boundary nodes.
		final double[] xGrid = spaceTimeDiscretization.getSpaceGrid(0).getGrid();
		final int nX = xGrid.length;

		final double theta = spaceTimeDiscretization.getTheta();

		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int M = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		// Derivative operators on full grid.
		final FiniteDifferenceMatrixBuilder fdBuilder = new FiniteDifferenceMatrixBuilder(xGrid);
		final RealMatrix T1 = fdBuilder.getFirstDerivativeMatrix();
		final RealMatrix T2 = fdBuilder.getSecondDerivativeMatrix();
		final RealMatrix I = MatrixUtils.createRealIdentityMatrix(nX);

		// Initial condition at maturity (tau = 0): payoff on full grid.
		RealMatrix U = MatrixUtils.createRealMatrix(nX, 1);
		for (int i = 0; i < nX; i++) {
			U.setEntry(i, 0, valueAtMaturity.applyAsDouble(xGrid[i]));
		}

		final RealMatrix z = MatrixUtils.createRealMatrix(nX, timeLength);
		z.setColumnMatrix(0, U);

		for (int m = 0; m < M; m++) {

			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			// Calendar time t (not time-to-maturity): align with existing solver convention.
			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(M - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(M - (m + 1));

			final double tSafe_m = (t_m == 0.0 ? 1e-6 : t_m);
			final double tSafe_mp1 = (t_mp1 == 0.0 ? 1e-6 : t_mp1);

			// Risk-free rates from discount curve (same convention as current solvers).
			final double r_m = -Math.log(model.getRiskFreeCurve().getDiscountFactor(tSafe_m)) / tSafe_m;
			final double r_mp1 = -Math.log(model.getRiskFreeCurve().getDiscountFactor(tSafe_mp1)) / tSafe_mp1;

			// Build coefficient diagonals on the grid:
			// mu(t,x): drift of the state variable X
			// a(t,x):  sum_k b_k(t,x)^2, where b_k are factor loadings of X
			final double[] mu_m = new double[nX];
			final double[] mu_mp1 = new double[nX];

			final double[] a_m = new double[nX];
			final double[] a_mp1 = new double[nX];

			for (int i = 0; i < nX; i++) {
				final double x = xGrid[i];

				mu_m[i] = model.getDrift(t_m, x)[0];
				mu_mp1[i] = model.getDrift(t_mp1, x)[0];

				final double[][] b_m = model.getFactorLoading(t_m, x);
				final double[][] b_mp1 = model.getFactorLoading(t_mp1, x);

				double am = 0.0;
				for (int f = 0; f < b_m[0].length; f++) {
					final double b = b_m[0][f];
					am += b * b;
				}

				double ap = 0.0;
				for (int f = 0; f < b_mp1[0].length; f++) {
					final double b = b_mp1[0][f];
					ap += b * b;
				}

				a_m[i] = am;
				a_mp1[i] = ap;
			}

			final RealMatrix Mu_m = new DiagonalMatrix(mu_m);
			final RealMatrix Mu_mp1 = new DiagonalMatrix(mu_mp1);
			final RealMatrix A_m = new DiagonalMatrix(a_m);
			final RealMatrix A_mp1 = new DiagonalMatrix(a_mp1);

			// Drift and diffusion terms.
			final RealMatrix driftTerm_m = Mu_m.scalarMultiply(deltaTau).multiply(T1);
			final RealMatrix driftTerm_mp1 = Mu_mp1.scalarMultiply(deltaTau).multiply(T1);

			final RealMatrix diffTerm_m = A_m.scalarMultiply(0.5 * deltaTau).multiply(T2);
			final RealMatrix diffTerm_mp1 = A_mp1.scalarMultiply(0.5 * deltaTau).multiply(T2);

			// F and G in theta scheme (mirrors FDMThetaMethod1D).
			final RealMatrix F = I.scalarMultiply(1.0 - r_m * deltaTau).add(driftTerm_m).add(diffTerm_m);
			final RealMatrix G = I.scalarMultiply(1.0 + r_mp1 * deltaTau).subtract(driftTerm_mp1).subtract(diffTerm_mp1);

			RealMatrix H = G.scalarMultiply(theta).add(I.scalarMultiply(1.0 - theta));
			final RealMatrix A = F.scalarMultiply(1.0 - theta).add(I.scalarMultiply(theta));

			RealMatrix rhs = A.multiply(U);

			// Dirichlet boundary enforcement (first and last grid point).
			final double tau_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);
			for (int i = 0; i < nX; i++) {
				if (i != 0 && i != nX - 1) {
					continue;
				}

				final double boundaryValue = (i == 0)
						? timeReversedLowerBoundary(xGrid[i], tau_mp1)
						: timeReversedUpperBoundary(xGrid[i], tau_mp1);

				for (int col = 0; col < nX; col++) {
					H.setEntry(i, col, 0.0);
				}
				H.setEntry(i, i, 1.0);
				rhs.setEntry(i, 0, boundaryValue);
			}

			if (exercise == ExerciseType.EUROPEAN) {
				final DecompositionSolver solver = new LUDecomposition(H).getSolver();
				U = solver.solve(rhs);
			}
			else if (exercise == ExerciseType.AMERICAN) {
				final double omega = 1.2;
				final SORDecomposition sor = new SORDecomposition(H);
				final RealMatrix zz = sor.getSol(U, rhs, omega, 500);

				for (int i = 0; i < nX; i++) {
					if (i == 0) {
						U.setEntry(i, 0, timeReversedLowerBoundary(xGrid[i], tau_mp1));
					}
					else if (i == nX - 1) {
						U.setEntry(i, 0, timeReversedUpperBoundary(xGrid[i], tau_mp1));
					}
					else {
						U.setEntry(i, 0, Math.max(zz.getEntry(i, 0), valueAtMaturity.applyAsDouble(xGrid[i])));
					}
				}
			}

			z.setColumnMatrix(m + 1, U);
		}

		return z.getData();
	}

	@Override
	public double[] getValue(
			final double evaluationTime,
			final double time,
			final DoubleUnaryOperator valueAtMaturity) {

		final RealMatrix values = new Array2DRowRealMatrix(getValues(time, valueAtMaturity));

		// tau = T - t
		final double tau = time - evaluationTime;
		final int timeIndex = this.spaceTimeDiscretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);

		return values.getColumn(timeIndex);
	}

	private double timeReversedLowerBoundary(final double x, final double tau) {
		return model.getValueAtLowerBoundary(
				product,
				spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tau,
				x)[0];
	}

	private double timeReversedUpperBoundary(final double x, final double tau) {
		return model.getValueAtUpperBoundary(
				product,
				spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tau,
				x)[0];
	}
}