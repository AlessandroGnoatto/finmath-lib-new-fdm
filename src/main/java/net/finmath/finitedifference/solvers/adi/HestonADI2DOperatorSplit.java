package net.finmath.finitedifference.solvers.adi;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.solvers.TridiagonalMatrix;

/**
 * Heston-oriented 2D semidiscrete operator split for ADI schemes.
 *
 * <p>
 * The operator is split as
 * </p>
 *
 * <pre>
 * A = A0 + A1 + A2
 * </pre>
 *
 * <p>
 * with
 * </p>
 * <ul>
 *   <li>{@code A0}: mixed derivative only,</li>
 *   <li>{@code A1}: first-direction drift + diffusion + half reaction,</li>
 *   <li>{@code A2}: second-direction drift + diffusion + half reaction.</li>
 * </ul>
 *
 * <p>
 * This follows the usual ADI organization used for Heston-type PDEs and is
 * intended as a cleaner alternative to burying the full reaction term in A0.
 * </p>
 *
 * <p>
 * This first version still uses the generic directional stencil builder for
 * A1 and A2. Heston-specific refinements near v = 0 can be added later inside
 * this class without changing the solver hierarchy.
 * </p>
 * 
 * @author Alessandro Gnoatto
 */
public class HestonADI2DOperatorSplit implements ADI2DOperatorSplit {

	private final FiniteDifferenceEquityModel model;
	private final double[] x0Grid;
	private final double[] x1Grid;

	private final int n0;
	private final int n1;
	private final int n;

	private final ADI2DStencilBuilder stencilBuilder;

	public HestonADI2DOperatorSplit(
			final FiniteDifferenceEquityModel model,
			final double[] x0Grid,
			final double[] x1Grid) {
		this.model = model;
		this.x0Grid = x0Grid;
		this.x1Grid = x1Grid;

		this.n0 = x0Grid.length;
		this.n1 = x1Grid.length;
		this.n = n0 * n1;

		this.stencilBuilder = new ADI2DStencilBuilder(model, x0Grid, x1Grid);
	}

	@Override
	public double[] applyA0(final double[] u, final double time) {
		checkLength(u);

		final double[] out = new double[n];

		for(int j = 1; j < n1 - 1; j++) {
			for(int i = 1; i < n0 - 1; i++) {
				final int k = flatten(i, j);

				final double x0 = x0Grid[i];
				final double x1 = x1Grid[j];

				final double[][] factorLoading = model.getFactorLoading(time, x0, x1);

				double a01 = 0.0;
				for(int f = 0; f < factorLoading[0].length; f++) {
					a01 += factorLoading[0][f] * factorLoading[1][f];
				}

				final double dx0Down = x0Grid[i] - x0Grid[i - 1];
				final double dx0Up = x0Grid[i + 1] - x0Grid[i];
				final double dx1Down = x1Grid[j] - x1Grid[j - 1];
				final double dx1Up = x1Grid[j + 1] - x1Grid[j];

				final double d0d1 =
						(
								u[flatten(i + 1, j + 1)]
								- u[flatten(i + 1, j - 1)]
								- u[flatten(i - 1, j + 1)]
								+ u[flatten(i - 1, j - 1)]
						)
						/ ((dx0Down + dx0Up) * (dx1Down + dx1Up));

				out[k] = a01 * d0d1;
			}
		}

		return out;
	}

	@Override
	public double[] applyA1(final double[] u, final double time) {
		checkLength(u);

		final double[] out = stencilBuilder.applyFirstDirectionOperator(u, time);
		final double halfRate = 0.5 * getShortRate(time);

		for(int k = 0; k < out.length; k++) {
			out[k] -= halfRate * u[k];
		}

		return out;
	}

	@Override
	public double[] applyA2(final double[] u, final double time) {
		checkLength(u);

		final double[] out = stencilBuilder.applySecondDirectionOperator(u, time);
		final double halfRate = 0.5 * getShortRate(time);

		for(int k = 0; k < out.length; k++) {
			out[k] -= halfRate * u[k];
		}

		return out;
	}

	@Override
	public double[] applyA(final double[] u, final double time) {
		return add(add(applyA0(u, time), applyA1(u, time)), applyA2(u, time));
	}

	@Override
	public TridiagonalMatrix buildFirstDirectionLineMatrix(
			final double time,
			final double dt,
			final double theta,
			final int secondDirectionIndex) {

		final TridiagonalMatrix matrix =
				stencilBuilder.buildFirstDirectionLineMatrix(time, dt, theta, secondDirectionIndex);

		final double halfRate = 0.5 * getShortRate(time);

		for(int i = 1; i < n0 - 1; i++) {
			matrix.diag[i] += theta * dt * halfRate;
		}

		return matrix;
	}

	@Override
	public TridiagonalMatrix buildSecondDirectionLineMatrix(
			final double time,
			final double dt,
			final double theta,
			final int firstDirectionIndex) {

		final TridiagonalMatrix matrix =
				stencilBuilder.buildSecondDirectionLineMatrix(time, dt, theta, firstDirectionIndex);

		final double halfRate = 0.5 * getShortRate(time);

		for(int j = 1; j < n1 - 1; j++) {
			matrix.diag[j] += theta * dt * halfRate;
		}

		return matrix;
	}

	protected double getShortRate(final double time) {
		final double tSafe = Math.max(time, 1E-10);
		final double discountFactor = model.getRiskFreeCurve().getDiscountFactor(tSafe);
		return -Math.log(discountFactor) / tSafe;
	}

	protected int flatten(final int i0, final int i1) {
		return i0 + i1 * n0;
	}

	private void checkLength(final double[] u) {
		if(u == null || u.length != n) {
			throw new IllegalArgumentException("State vector has wrong length.");
		}
	}

	private double[] add(final double[] a, final double[] b) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = a[i] + b[i];
		}
		return out;
	}
}