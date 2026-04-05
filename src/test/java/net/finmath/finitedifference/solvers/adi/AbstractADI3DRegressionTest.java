package net.finmath.finitedifference.solvers.adi;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.function.DoubleUnaryOperator;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.boundaries.StandardBoundaryCondition;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Minimal regression test for AbstractADI3D.
 *
 * <p>
 * This test uses a zero-operator 3D model. A constant terminal condition should
 * remain constant under backward time stepping.
 * </p>
 */
public class AbstractADI3DRegressionTest {

	@Test
	public void testFlattenUnflattenConsistency() {
		final int n0 = 4;
		final int n1 = 5;
		final int n2 = 3;

		for(int i2 = 0; i2 < n2; i2++) {
			for(int i1 = 0; i1 < n1; i1++) {
				for(int i0 = 0; i0 < n0; i0++) {
					final int flat = FDM3DGridUtil.flatten(i0, i1, i2, n0, n1);
					final int[] idx = FDM3DGridUtil.unflatten(flat, n0, n1, n2);

					assertEquals(i0, idx[0]);
					assertEquals(i1, idx[1]);
					assertEquals(i2, idx[2]);
				}
			}
		}
	}

	@Test
	public void testConstantSolutionRemainsConstant() {
		final Grid x0 = new UniformGrid(4, 0.0, 1.0);
		final Grid x1 = new UniformGrid(4, 0.0, 1.0);
		final Grid x2 = new UniformGrid(4, 0.0, 1.0);

		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(0.0, 5, 0.2);

		final SpaceTimeDiscretization spaceTimeDiscretization = new SpaceTimeDiscretization(
				new Grid[] { x0, x1, x2 },
				timeDiscretization,
				0.5,
				new double[] { 0.5, 0.5, 0.5 });

		final FiniteDifferenceEquityModel model = new ZeroOperator3DModel(spaceTimeDiscretization);
		final FiniteDifferenceProduct product = new DummyProduct();

		final AbstractADI3D solver = new IdentityLineSolveADI3D(
				model,
				product,
				spaceTimeDiscretization,
				new EuropeanExercise(1.0));

		final DoubleUnaryOperator terminalValue = x -> 7.5;

		final double[][] values = solver.getValues(1.0, terminalValue);

		final int expectedSpatialPoints =
				x0.getGrid().length * x1.getGrid().length * x2.getGrid().length;
		final int expectedTimePoints = timeDiscretization.getNumberOfTimeSteps() + 1;

		assertEquals(expectedSpatialPoints, values.length);
		assertEquals(expectedTimePoints, values[0].length);

		for(int k = 0; k < values.length; k++) {
			for(int t = 0; t < values[k].length; t++) {
				assertTrue(Double.isFinite(values[k][t]));
				assertEquals(7.5, values[k][t], 1E-12);
			}
		}
	}

	private static final class IdentityLineSolveADI3D extends AbstractADI3D {

		private IdentityLineSolveADI3D(
				final FiniteDifferenceEquityModel model,
				final FiniteDifferenceProduct product,
				final SpaceTimeDiscretization spaceTimeDiscretization,
				final EuropeanExercise exercise) {
			super(model, product, spaceTimeDiscretization, exercise);
		}

		@Override
		protected double[] solveFirstDirectionLines(
				final double[] rhs,
				final double time,
				final double dt) {
			return rhs.clone();
		}

		@Override
		protected double[] solveSecondDirectionLines(
				final double[] rhs,
				final double time,
				final double dt) {
			return rhs.clone();
		}

		@Override
		protected double[] solveThirdDirectionLines(
				final double[] rhs,
				final double time,
				final double dt) {
			return rhs.clone();
		}
	}

	private static final class DummyProduct implements FiniteDifferenceProduct {

		@Override
		public double[] getValue(final double evaluationTime, final FiniteDifferenceEquityModel model) {
			throw new UnsupportedOperationException("Not needed for this test.");
		}

		@Override
		public double[][] getValues(final FiniteDifferenceEquityModel model) {
			throw new UnsupportedOperationException("Not needed for this test.");
		}
	}

	private static final class ZeroOperator3DModel implements FiniteDifferenceEquityModel {

		private final SpaceTimeDiscretization spaceTimeDiscretization;
		private final DiscountCurve discountCurve;

		private ZeroOperator3DModel(final SpaceTimeDiscretization spaceTimeDiscretization) {
			this.spaceTimeDiscretization = spaceTimeDiscretization;
			this.discountCurve = DiscountCurveInterpolation.createDiscountCurveFromDiscountFactors(
					"flat-discount",
					new double[] { 0.0, 1.0 },
					new double[] { 1.0, 1.0 });
		}

		@Override
		public DiscountCurve getRiskFreeCurve() {
			return discountCurve;
		}

		@Override
		public DiscountCurve getDividendYieldCurve() {
			return discountCurve;
		}

		@Override
		public SpaceTimeDiscretization getSpaceTimeDiscretization() {
			return spaceTimeDiscretization;
		}

		@Override
		public double[] getDrift(final double time, final double... stateVariables) {
			return new double[] { 0.0, 0.0, 0.0 };
		}

		@Override
		public double[][] getFactorLoading(final double time, final double... stateVariables) {
			return new double[][] {
				{ 0.0 },
				{ 0.0 },
				{ 0.0 }
			};
		}

		@Override
		public double[] getInitialValue() {
			return new double[] { 0.5, 0.5, 0.5 };
		}

		@Override
		public BoundaryCondition[] getBoundaryConditionsAtLowerBoundary(
				final FiniteDifferenceProduct product,
				final double time,
				final double... stateVariables) {
			return new BoundaryCondition[] {
					StandardBoundaryCondition.none(),
					StandardBoundaryCondition.none(),
					StandardBoundaryCondition.none()
			};
		}

		@Override
		public BoundaryCondition[] getBoundaryConditionsAtUpperBoundary(
				final FiniteDifferenceProduct product,
				final double time,
				final double... stateVariables) {
			return new BoundaryCondition[] {
					StandardBoundaryCondition.none(),
					StandardBoundaryCondition.none(),
					StandardBoundaryCondition.none()
			};
		}

		@Override
		public FiniteDifferenceEquityModel getCloneWithModifiedSpaceTimeDiscretization(
				final SpaceTimeDiscretization newSpaceTimeDiscretization) {
			return new ZeroOperator3DModel(newSpaceTimeDiscretization);
		}
	}
}