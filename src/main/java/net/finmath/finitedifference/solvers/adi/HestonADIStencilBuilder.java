package net.finmath.finitedifference.solvers.adi;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.solvers.TridiagonalMatrix;

/**
 * Builder for the tridiagonal line matrices arising in alternating direction
 * implicit (ADI) finite difference solves for the two-dimensional Heston PDE.
 * <p>
 * The state variables are:
 * </p>
 * <ul>
 *   <li>state variable {@code 0}: spot {@code S},</li>
 *   <li>state variable {@code 1}: variance {@code v}.</li>
 * </ul>
 *
 * <p>
 * In the ADI splitting used by the solver, the differential operator is
 * decomposed into directional parts:
 * </p>
 * <ul>
 *   <li>{@code A1}: spot-direction drift and diffusion terms,</li>
 *   <li>{@code A2}: variance-direction drift and diffusion terms.</li>
 * </ul>
 *
 * <p>
 * This class constructs the tridiagonal matrices corresponding to the implicit
 * one-dimensional line solves
 * </p>
 * <pre>
 * (I - theta * dt * A1)
 * </pre>
 * <p>
 * for a fixed variance level, and
 * </p>
 * <pre>
 * (I - theta * dt * A2)
 * </pre>
 * <p>
 * for a fixed spot level.
 * </p>
 *
 * <p>
 * The coefficients are assembled from the local drift and factor loading
 * returned by the {@link FDMHestonModel}. Spatial derivatives are discretized
 * by central finite differences on possibly non-uniform grids. As a result,
 * each directional operator gives rise to a tridiagonal matrix on the
 * corresponding line.
 * </p>
 *
 * <p>
 * The boundary rows are initialized in a simple identity form and are intended
 * to be overwritten by the calling solver when enforcing the appropriate
 * boundary conditions.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class HestonADIStencilBuilder {

	private final FDMHestonModel model;
	private final double[] sGrid;
	private final double[] vGrid;

	/**
	 * Creates a stencil builder for ADI line solves under a Heston model.
	 *
	 * @param model The finite difference Heston model providing drift and factor
	 * 		loadings.
	 * @param sGrid The spatial grid for the spot state variable.
	 * @param vGrid The spatial grid for the variance state variable.
	 */
	public HestonADIStencilBuilder(
			final FDMHestonModel model,
			final double[] sGrid,
			final double[] vGrid) {
		this.model = model;
		this.sGrid = sGrid;
		this.vGrid = vGrid;
	}

	/**
	 * Builds the tridiagonal matrix for the implicit spot-direction solve on a
	 * fixed variance slice.
	 * <p>
	 * The returned matrix represents the operator
	 * </p>
	 * <pre>
	 * (I - theta * dt * A1)
	 * </pre>
	 * <p>
	 * where {@code A1} contains the spot-direction drift and diffusion terms
	 * only. The variance level is kept fixed at the value corresponding to the
	 * provided variance-grid index.
	 * </p>
	 *
	 * <p>
	 * For each interior spot node, the operator coefficients are obtained from:
	 * </p>
	 * <ul>
	 *   <li>the spot drift component,</li>
	 *   <li>the spot diffusion coefficient derived from the factor loading,</li>
	 *   <li>central finite difference approximations on the possibly non-uniform
	 *       spot grid.</li>
	 * </ul>
	 *
	 * <p>
	 * The first and last rows are initialized as identity rows. They are intended
	 * to be replaced by the caller when applying the actual boundary conditions.
	 * </p>
	 *
	 * @param time The running time at which the local operator coefficients are evaluated.
	 * @param dt The time step size used in the ADI scheme.
	 * @param theta The ADI weighting parameter.
	 * @param vIndex The index of the fixed variance level.
	 * @return The tridiagonal matrix representing the implicit spot-direction line solve.
	 */
	public TridiagonalMatrix buildSpotLineMatrix(
			final double time,
			final double dt,
			final double theta,
			final int vIndex) {

		final int nS = sGrid.length;
		final double v = vGrid[vIndex];
		final TridiagonalMatrix m = new TridiagonalMatrix(nS);

		/*
		 * Boundary rows are overwritten by the caller.
		 */
		m.diag[0] = 1.0;
		m.diag[nS - 1] = 1.0;

		for(int i = 1; i < nS - 1; i++) {
			final double s = sGrid[i];

			final double[] drift = model.getDrift(time, s, v);
			final double[][] factorLoading = model.getFactorLoading(time, s, v);

			final double muS = drift[0];

			double aSS = 0.0;
			for(int f = 0; f < factorLoading[0].length; f++) {
				aSS += factorLoading[0][f] * factorLoading[0][f];
			}

			final double dsDown = sGrid[i] - sGrid[i - 1];
			final double dsUp = sGrid[i + 1] - sGrid[i];
			final double dsSum = dsDown + dsUp;
			final double dsProd = dsDown * dsUp;

			/*
			 * Non-uniform central differences:
			 *
			 * d/dS    ~ (u_{i+1} - u_{i-1}) / (dsDown + dsUp)
			 *
			 * d2/dS2  ~ 2/(dsDown+dsUp) * [ (u_{i+1}-u_i)/dsUp - (u_i-u_{i-1})/dsDown ]
			 *
			 * Hence A1 u = lower*u_{i-1} + diag*u_i + upper*u_{i+1} with
			 *
			 * lower = -muS/dsSum + aSS/(dsSum*dsDown)
			 * diag  = -aSS/dsProd
			 * upper =  muS/dsSum + aSS/(dsSum*dsUp)
			 *
			 * and the implicit line solve is
			 *
			 * (I - theta*dt*A1)
			 */
			final double lowerA1 = -muS / dsSum + aSS / (dsSum * dsDown);
			final double diagA1  = -aSS / dsProd;
			final double upperA1 =  muS / dsSum + aSS / (dsSum * dsUp);

			m.lower[i] = -theta * dt * lowerA1;
			m.diag[i]  = 1.0 - theta * dt * diagA1;
			m.upper[i] = -theta * dt * upperA1;
		}

		return m;
	}

	/**
	 * Builds the tridiagonal matrix for the implicit variance-direction solve on
	 * a fixed spot slice.
	 * <p>
	 * The returned matrix represents the operator
	 * </p>
	 * <pre>
	 * (I - theta * dt * A2)
	 * </pre>
	 * <p>
	 * where {@code A2} contains the variance-direction drift and diffusion terms
	 * only. The spot level is kept fixed at the value corresponding to the
	 * provided spot-grid index.
	 * </p>
	 *
	 * <p>
	 * For each interior variance node, the operator coefficients are obtained from:
	 * </p>
	 * <ul>
	 *   <li>the variance drift component,</li>
	 *   <li>the variance diffusion coefficient derived from the factor loading,</li>
	 *   <li>central finite difference approximations on the possibly non-uniform
	 *       variance grid.</li>
	 * </ul>
	 *
	 * <p>
	 * The first and last rows are initialized as identity rows. They are intended
	 * to be replaced by the caller when applying the actual boundary conditions.
	 * </p>
	 *
	 * @param time The running time at which the local operator coefficients are evaluated.
	 * @param dt The time step size used in the ADI scheme.
	 * @param theta The ADI weighting parameter.
	 * @param sIndex The index of the fixed spot level.
	 * @return The tridiagonal matrix representing the implicit variance-direction line solve.
	 */
	public TridiagonalMatrix buildVarianceLineMatrix(
			final double time,
			final double dt,
			final double theta,
			final int sIndex) {

		final int nV = vGrid.length;
		final double s = sGrid[sIndex];
		final TridiagonalMatrix m = new TridiagonalMatrix(nV);

		/*
		 * Boundary rows are overwritten by the caller.
		 */
		m.diag[0] = 1.0;
		m.diag[nV - 1] = 1.0;

		for(int j = 1; j < nV - 1; j++) {
			final double v = vGrid[j];

			final double[] drift = model.getDrift(time, s, v);
			final double[][] factorLoading = model.getFactorLoading(time, s, v);

			final double muV = drift[1];

			double aVV = 0.0;
			for(int f = 0; f < factorLoading[1].length; f++) {
				aVV += factorLoading[1][f] * factorLoading[1][f];
			}

			final double dvDown = vGrid[j] - vGrid[j - 1];
			final double dvUp = vGrid[j + 1] - vGrid[j];
			final double dvSum = dvDown + dvUp;
			final double dvProd = dvDown * dvUp;

			/*
			 * Same non-uniform central-difference logic as in spot direction.
			 *
			 * A2 u = lower*u_{j-1} + diag*u_j + upper*u_{j+1}
			 */
			final double lowerA2 = -muV / dvSum + aVV / (dvSum * dvDown);
			final double diagA2  = -aVV / dvProd;
			final double upperA2 =  muV / dvSum + aVV / (dvSum * dvUp);

			m.lower[j] = -theta * dt * lowerA2;
			m.diag[j]  = 1.0 - theta * dt * diagA2;
			m.upper[j] = -theta * dt * upperA2;
		}

		return m;
	}
}