package net.finmath.finitedifference.solvers;

import java.util.function.DoubleUnaryOperator;

/**
 * Interface for finite difference solvers.
 *
 * <p>
 * Implementations provide methods to compute the solution of a PDE
 * on a space-time grid, either at a specific evaluation time or
 * over the full time history.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public interface FDMSolver {

	/**
	 * Returns the solution at a given evaluation time.
	 *
	 * <p>
	 * This is a legacy return type. Typical shapes are:
	 * </p>
	 * <ul>
	 *   <li>1D: {@code [nS]} representing values at the evaluation time.</li>
	 *   <li>2D: {@code [nS * nV]} representing values at the evaluation time.</li>
	 * </ul>
	 *
	 * @param evaluationTime   The evaluation time.
	 * @param time             The maturity (time to maturity).
	 * @param valueAtMaturity  The payoff function applied at maturity.
	 * @return The solution at the specified evaluation time.
	 */
	double[] getValue(
			double evaluationTime,
			double time,
			DoubleUnaryOperator valueAtMaturity);

	/**
	 * Returns the full time history of the solution on the space-time grid.
	 *
	 * <p>
	 * Typical shapes are:
	 * </p>
	 * <ul>
	 *   <li>1D: {@code [nT][nS]}</li>
	 *   <li>2D: {@code [nT][nS * nV]}</li>
	 * </ul>
	 *
	 * @param time            The maturity (time to maturity).
	 * @param valueAtMaturity The payoff function applied at maturity.
	 * @return The full time-space solution.
	 */
	double[][] getValues(
			double time,
			DoubleUnaryOperator valueAtMaturity);
}