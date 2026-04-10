package net.finmath.finitedifference.solvers.adi;

import net.finmath.finitedifference.solvers.TridiagonalMatrix;

/**
 * Semidiscrete operator split for 2D ADI schemes.
 *
 * <p>
 * The full semidiscrete operator is written as
 * </p>
 *
 * <pre>
 * A = A0 + A1 + A2
 * </pre>
 *
 * <p>
 * where
 * </p>
 * <ul>
 *   <li>{@code A0} is the mixed / explicitly treated part,</li>
 *   <li>{@code A1} is the first-direction part,</li>
 *   <li>{@code A2} is the second-direction part.</li>
 * </ul>
 *
 * <p>
 * The interface also provides the line matrices corresponding to the implicit
 * directional operators used by ADI schemes.
 * </p>
 * 
 * @author Alessandro Gnoatto
 */
public interface ADI2DOperatorSplit {

	/**
	 * Applies the explicitly treated part A0.
	 *
	 * @param u Flattened state vector.
	 * @param time Running time.
	 * @return A0 u
	 */
	double[] applyA0(double[] u, double time);

	/**
	 * Applies the first-direction part A1.
	 *
	 * @param u Flattened state vector.
	 * @param time Running time.
	 * @return A1 u
	 */
	double[] applyA1(double[] u, double time);

	/**
	 * Applies the second-direction part A2.
	 *
	 * @param u Flattened state vector.
	 * @param time Running time.
	 * @return A2 u
	 */
	double[] applyA2(double[] u, double time);

	/**
	 * Applies the full operator A = A0 + A1 + A2.
	 *
	 * @param u Flattened state vector.
	 * @param time Running time.
	 * @return A u
	 */
	double[] applyA(double[] u, double time);

	/**
	 * Builds the tridiagonal matrix corresponding to
	 *
	 * <pre>
	 * I - theta * dt * A1
	 * </pre>
	 *
	 * on one fixed slice of the second state variable.
	 *
	 * @param time Running time.
	 * @param dt Time step.
	 * @param theta ADI weight.
	 * @param secondDirectionIndex Fixed index in second state variable.
	 * @return Tridiagonal implicit first-direction matrix.
	 */
	TridiagonalMatrix buildFirstDirectionLineMatrix(double time, double dt, double theta, int secondDirectionIndex);

	/**
	 * Builds the tridiagonal matrix corresponding to
	 *
	 * <pre>
	 * I - theta * dt * A2
	 * </pre>
	 *
	 * on one fixed slice of the first state variable.
	 *
	 * @param time Running time.
	 * @param dt Time step.
	 * @param theta ADI weight.
	 * @param firstDirectionIndex Fixed index in first state variable.
	 * @return Tridiagonal implicit second-direction matrix.
	 */
	TridiagonalMatrix buildSecondDirectionLineMatrix(double time, double dt, double theta, int firstDirectionIndex);
}