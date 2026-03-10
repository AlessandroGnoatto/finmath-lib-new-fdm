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
 * Constructs and solves matrix systems for the theta method applied to
 * one-dimensional local volatility PDEs using finite differences.
 *
 * <p>
 * This class implements the theta method for time discretization, allowing
 * interpolation between explicit and implicit schemes depending on the theta
 * parameter. It is designed to handle local volatility models with generic
 * spatial grids, supporting both uniform and non-uniform grid spacing.
 * </p>
 *
 * <p>
 * The implementation supports the pricing of both European and American options.
 * </p>
 *
 * @author Ralph Rudd
 * @author Christian Fries
 * @author Jörg Kienitz
 * @author Alessandro Gnoatto
 */
public class FDMThetaMethod1D implements FDMSolver {

	private final FiniteDifferenceEquityModel model;
	private final FiniteDifferenceProduct product;
	private final SpaceTimeDiscretization spaceTimeDiscretization;
	private final ExerciseType exercise;

	/**
	 * Creates a theta-method solver for a one-dimensional PDE.
	 *
	 * @param model                  The finite difference equity model.
	 * @param product                The finite difference product.
	 * @param spaceTimeDiscretization The space-time discretization.
	 * @param exercise               The exercise type (European or American).
	 */
	public FDMThetaMethod1D(
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

		// Full grid including boundary nodes (aligns with FDMThetaMethod2D)
		final double[] stockGrid = spaceTimeDiscretization.getSpaceGrid(0).getGrid();
		final int N = stockGrid.length;
		final double theta = spaceTimeDiscretization.getTheta();

		final int timeLength = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int M = spaceTimeDiscretization.getTimeDiscretization().getNumberOfTimeSteps();

		// Derivative operators on full grid
		final FiniteDifferenceMatrixBuilder fdBuilder = new FiniteDifferenceMatrixBuilder(stockGrid);
		final RealMatrix T1 = fdBuilder.getFirstDerivativeMatrix();
		final RealMatrix T2 = fdBuilder.getSecondDerivativeMatrix();

		// Diagonal matrices for S and S^2
		final double[] s = new double[N];
		final double[] s2 = new double[N];
		for(int i = 0; i < N; i++) {
			s[i] = stockGrid[i];
			s2[i] = stockGrid[i] * stockGrid[i];
		}
		final RealMatrix D1 = new DiagonalMatrix(s);
		final RealMatrix D2 = new DiagonalMatrix(s2);
		final RealMatrix I = MatrixUtils.createRealIdentityMatrix(N);

		// Initial condition at maturity (tau = 0): payoff on full grid
		RealMatrix U = MatrixUtils.createRealMatrix(N, 1);
		for(int i = 0; i < N; i++) {
			U.setEntry(i, 0, valueAtMaturity.applyAsDouble(stockGrid[i]));
		}

		final RealMatrix z = MatrixUtils.createRealMatrix(N, timeLength);
		z.setColumnMatrix(0, U);

		for(int m = 0; m < M; m++) {
			final double deltaTau = spaceTimeDiscretization.getTimeDiscretization().getTimeStep(m);

			// We call curves/coefficients with calendar time t (not time-to-maturity)
			final double t_m = spaceTimeDiscretization.getTimeDiscretization().getTime(M - m);
			final double t_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(M - (m + 1));

			final double drift_m = model.getDrift(t_m, stockGrid[0])[0];
			final double drift_mp1 = model.getDrift(t_mp1, stockGrid[0])[0];

			final double tSafe_m = (t_m == 0.0 ? 1e-6 : t_m);
			final double tSafe_mp1 = (t_mp1 == 0.0 ? 1e-6 : t_mp1);
			final double r_m = -Math.log(model.getRiskFreeCurve().getDiscountFactor(tSafe_m)) / tSafe_m;
			final double r_mp1 = -Math.log(model.getRiskFreeCurve().getDiscountFactor(tSafe_mp1)) / tSafe_mp1;

			final double[] sigma_m = new double[N];
			final double[] sigma_mp1 = new double[N];
			for(int i = 0; i < N; i++) {
				sigma_m[i] = Math.pow(model.getFactorLoading(t_m, stockGrid[i])[0][0], 2);
				sigma_mp1[i] = Math.pow(model.getFactorLoading(t_mp1, stockGrid[i])[0][0], 2);
			}
			final RealMatrix Sigma_m = new DiagonalMatrix(sigma_m);
			final RealMatrix Sigma_mp1 = new DiagonalMatrix(sigma_mp1);

			final RealMatrix driftTerm_m = D1.scalarMultiply(drift_m * deltaTau).multiply(T1);
			final RealMatrix driftTerm_mp1 = D1.scalarMultiply(drift_mp1 * deltaTau).multiply(T1);
			final RealMatrix diffBase = D2.scalarMultiply(0.5 * deltaTau).multiply(T2);
			final RealMatrix diffTerm_m = Sigma_m.multiply(diffBase);
			final RealMatrix diffTerm_mp1 = Sigma_mp1.multiply(diffBase);

			final RealMatrix F = I.scalarMultiply(1.0 - r_m * deltaTau).add(driftTerm_m).add(diffTerm_m);
			final RealMatrix G = I.scalarMultiply(1.0 + r_mp1 * deltaTau).subtract(driftTerm_mp1).subtract(diffTerm_mp1);

			RealMatrix H = G.scalarMultiply(theta).add(I.scalarMultiply(1.0 - theta));
			final RealMatrix A = F.scalarMultiply(1.0 - theta).add(I.scalarMultiply(theta));
			RealMatrix rhs = A.multiply(U);

			// Dirichlet boundary enforcement: overwrite rows in H and rhs
			final double tau_mp1 = spaceTimeDiscretization.getTimeDiscretization().getTime(m + 1);
			for(int i = 0; i < N; i++) {
				if(i != 0 && i != N - 1) {
					continue;
				}
				final double boundaryValue = (i == 0)
						? timeReversedLowerBoundary(stockGrid[i], tau_mp1)
						: timeReversedUpperBoundary(stockGrid[i], tau_mp1);

				for(int col = 0; col < N; col++) {
					H.setEntry(i, col, 0.0);
				}
				H.setEntry(i, i, 1.0);
				rhs.setEntry(i, 0, boundaryValue);
			}

			if(exercise == ExerciseType.EUROPEAN) {
				final DecompositionSolver solver = new LUDecomposition(H).getSolver();
				U = solver.solve(rhs);
			}
			else if(exercise == ExerciseType.AMERICAN) {
				final double omega = 1.2;
				final SORDecomposition sor = new SORDecomposition(H);
				final RealMatrix zz = sor.getSol(U, rhs, omega, 500);

				for(int i = 0; i < N; i++) {
					if(i == 0) {
						U.setEntry(i, 0, timeReversedLowerBoundary(stockGrid[i], tau_mp1));
					}
					else if(i == N - 1) {
						U.setEntry(i, 0, timeReversedUpperBoundary(stockGrid[i], tau_mp1));
					}
					else {
						U.setEntry(
								i,
								0,
								Math.max(zz.getEntry(i, 0), valueAtMaturity.applyAsDouble(stockGrid[i])));
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
		final int timeIndex =
				this.spaceTimeDiscretization.getTimeDiscretization()
						.getTimeIndexNearestLessOrEqual(tau);

		return values.getColumn(timeIndex);
	}

	private double timeReversedLowerBoundary(final double stockPrice, final double tau) {
		return model.getValueAtLowerBoundary(
				product,
				spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tau,
				stockPrice)[0];
	}

	private double timeReversedUpperBoundary(final double stockPrice, final double tau) {
		return model.getValueAtUpperBoundary(
				product,
				spaceTimeDiscretization.getTimeDiscretization().getLastTime() - tau,
				stockPrice)[0];
	}
}