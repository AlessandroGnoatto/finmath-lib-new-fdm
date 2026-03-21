package net.finmath.finitedifference.solvers.adi;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;

/**
 * Builds tridiagonal line matrices for ADI solves under the Heston model.
 *
 * <p>
 * State variables:
 * </p>
 * <ul>
 *   <li>state 0 = spot S</li>
 *   <li>state 1 = variance v</li>
 * </ul>
 *
 * <p>
 * We split the PDE operator into:
 * </p>
 * <ul>
 *   <li>A1 = spot-direction drift + diffusion</li>
 *   <li>A2 = variance-direction drift + diffusion</li>
 * </ul>
 *
 * <p>
 * This builder returns the tridiagonal matrix for
 * </p>
 * <p>
 * (I - theta * dt * A1)
 * </p>
 * <p>
 * or
 * </p>
 * <p>
 * (I - theta * dt * A2)
 * </p>
 * <p>
 * on one fixed line.
 * </p>
 */
public class HestonADIStencilBuilder {

	private final FDMHestonModel model;
	private final double[] sGrid;
	private final double[] vGrid;

	public HestonADIStencilBuilder(
			final FDMHestonModel model,
			final double[] sGrid,
			final double[] vGrid) {
		this.model = model;
		this.sGrid = sGrid;
		this.vGrid = vGrid;
	}

	/**
	 * Builds the tridiagonal matrix for the spot-direction implicit solve
	 * on a fixed variance slice:
	 *
	 * <p>
	 * (I - theta * dt * A1)
	 * </p>
	 *
	 * where A1 contains only spot drift and spot diffusion terms.
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
	 * Builds the tridiagonal matrix for the variance-direction implicit solve
	 * on a fixed spot slice:
	 *
	 * <p>
	 * (I - theta * dt * A2)
	 * </p>
	 *
	 * where A2 contains only variance drift and variance diffusion terms.
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