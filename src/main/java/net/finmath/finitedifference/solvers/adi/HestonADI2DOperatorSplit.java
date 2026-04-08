package net.finmath.finitedifference.solvers.adi;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.solvers.TridiagonalMatrix;

/**
 * Semidiscrete Heston operator split for 2D ADI schemes.
 *
 * <p>
 * The operator is split as
 * </p>
 * <pre>
 * A = A0 + A1 + A2
 * </pre>
 * <p>
 * where
 * </p>
 * <ul>
 *   <li>A0 = mixed derivative term,</li>
 *   <li>A1 = first-direction drift + diffusion + half reaction,</li>
 *   <li>A2 = second-direction drift + diffusion + half reaction.</li>
 * </ul>
 *
 * <p>
 * This mirrors the operator-split philosophy used in Rouah's Heston ADI code,
 * but preserves the line-solver architecture used in this Java code base.
 * </p>
 */
public class HestonADI2DOperatorSplit {

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

	/**
	 * Mixed derivative contribution only.
	 */
	public double[] applyA0(final double[] u, final double time) {
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

	/**
	 * First-direction drift + diffusion + half reaction.
	 */
	public double[] applyA1(final double[] u, final double time) {
		final double[] out = stencilBuilder.applyFirstDirectionOperator(u, time);

		final double tSafe = Math.max(time, 1E-10);
		final double discountFactor = model.getRiskFreeCurve().getDiscountFactor(tSafe);
		final double r = -Math.log(discountFactor) / tSafe;

		for(int k = 0; k < out.length; k++) {
			out[k] -= 0.5 * r * u[k];
		}

		return out;
	}

	/**
	 * Second-direction drift + diffusion + half reaction.
	 */
	public double[] applyA2(final double[] u, final double time) {
		final double[] out = stencilBuilder.applySecondDirectionOperator(u, time);

		final double tSafe = Math.max(time, 1E-10);
		final double discountFactor = model.getRiskFreeCurve().getDiscountFactor(tSafe);
		final double r = -Math.log(discountFactor) / tSafe;

		for(int k = 0; k < out.length; k++) {
			out[k] -= 0.5 * r * u[k];
		}

		return out;
	}

	/**
	 * Full operator A0 + A1 + A2.
	 */
	public double[] applyA(final double[] u, final double time) {
		return add(add(applyA0(u, time), applyA1(u, time)), applyA2(u, time));
	}

	/**
	 * Builds the first-direction implicit matrix corresponding to
	 *
	 * <pre>
	 * I - theta * dt * A1
	 * </pre>
	 *
	 * including half the reaction term on the diagonal.
	 */
	public TridiagonalMatrix buildFirstDirectionLineMatrix(
			final double time,
			final double dt,
			final double theta,
			final int x1Index) {

		final TridiagonalMatrix m = stencilBuilder.buildFirstDirectionLineMatrix(time, dt, theta, x1Index);

		final double tSafe = Math.max(time, 1E-10);
		final double discountFactor = model.getRiskFreeCurve().getDiscountFactor(tSafe);
		final double r = -Math.log(discountFactor) / tSafe;

		for(int i = 1; i < n0 - 1; i++) {
			m.diag[i] += theta * dt * 0.5 * r;
		}

		return m;
	}

	/**
	 * Builds the second-direction implicit matrix corresponding to
	 *
	 * <pre>
	 * I - theta * dt * A2
	 * </pre>
	 *
	 * including half the reaction term on the diagonal.
	 */
	public TridiagonalMatrix buildSecondDirectionLineMatrix(
			final double time,
			final double dt,
			final double theta,
			final int x0Index) {

		final TridiagonalMatrix m = stencilBuilder.buildSecondDirectionLineMatrix(time, dt, theta, x0Index);

		final double tSafe = Math.max(time, 1E-10);
		final double discountFactor = model.getRiskFreeCurve().getDiscountFactor(tSafe);
		final double r = -Math.log(discountFactor) / tSafe;

		for(int j = 1; j < n1 - 1; j++) {
			m.diag[j] += theta * dt * 0.5 * r;
		}

		return m;
	}

	public ADI2DStencilBuilder getStencilBuilder() {
		return stencilBuilder;
	}

	private int flatten(final int i0, final int i1) {
		return i0 + i1 * n0;
	}

	private double[] add(final double[] a, final double[] b) {
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			out[i] = a[i] + b[i];
		}
		return out;
	}
}