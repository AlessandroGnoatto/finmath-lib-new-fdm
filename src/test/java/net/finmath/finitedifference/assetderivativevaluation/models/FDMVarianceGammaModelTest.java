package net.finmath.finitedifference.assetderivativevaluation.models;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.Optional;

import org.junit.Test;

import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Unit tests for {@link FDMVarianceGammaModel}.
 *
 * @author Alessandro Gnoatto
 */
public class FDMVarianceGammaModelTest {

	private static final double INITIAL_VALUE = 100.0;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD_RATE = 0.02;

	private static final double SIGMA = 0.30;
	private static final double NU = 0.20;
	private static final double THETA_PARAMETER = -0.10;

	private static final double LOWER_INTEGRATION_BOUND = -2.0;
	private static final double UPPER_INTEGRATION_BOUND = 2.0;

	private static final double THETA = 0.5;

	private static final double TOL = 1E-12;

	@Test
	public void testInitialValueAndLocalCoefficients() {
		final SpaceTimeDiscretization spaceTimeDiscretization = createSpaceTimeDiscretization(10, 80);

		final FDMVarianceGammaModel model = new FDMVarianceGammaModel(
				INITIAL_VALUE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_RATE,
				SIGMA,
				NU,
				THETA_PARAMETER,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				spaceTimeDiscretization
		);

		assertArrayEquals(new double[] { INITIAL_VALUE }, model.getInitialValue(), TOL);

		final double time = 1.0;
		final double stateVariable = 120.0;

		final double[] drift = model.getDrift(time, stateVariable);

		final double riskFreeRateFromCurve =
				-Math.log(model.getRiskFreeCurve().getDiscountFactor(time)) / time;
		final double dividendYieldRateFromCurve =
				-Math.log(model.getDividendYieldCurve().getDiscountFactor(time)) / time;

		final double expectedDrift =
				(riskFreeRateFromCurve - dividendYieldRateFromCurve) * stateVariable;

		assertEquals(expectedDrift, drift[0], TOL);

		final double[][] factorLoading = model.getFactorLoading(time, stateVariable);
		assertEquals(1, factorLoading.length);
		assertEquals(1, factorLoading[0].length);
		assertEquals(0.0, factorLoading[0][0], TOL);
	}

	@Test
	public void testJumpComponentPresenceAndParameters() {
		final SpaceTimeDiscretization spaceTimeDiscretization = createSpaceTimeDiscretization(10, 80);

		final FDMVarianceGammaModel model = new FDMVarianceGammaModel(
				INITIAL_VALUE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_RATE,
				SIGMA,
				NU,
				THETA_PARAMETER,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				spaceTimeDiscretization
		);

		final Optional<JumpComponent> optionalJumpComponent = model.getJumpComponent();

		assertTrue(optionalJumpComponent.isPresent());
		assertTrue(optionalJumpComponent.get() instanceof VarianceGammaJumpComponent);

		final VarianceGammaJumpComponent jumpComponent =
				(VarianceGammaJumpComponent) optionalJumpComponent.get();

		assertEquals(LOWER_INTEGRATION_BOUND, jumpComponent.getLowerIntegrationBound(0.5, INITIAL_VALUE), TOL);
		assertEquals(UPPER_INTEGRATION_BOUND, jumpComponent.getUpperIntegrationBound(0.5, INITIAL_VALUE), TOL);

		final double sigmaSquared = SIGMA * SIGMA;
		final double radical = Math.sqrt(THETA_PARAMETER * THETA_PARAMETER + 2.0 * sigmaSquared / NU);

		final double expectedC = 1.0 / NU;
		final double expectedG = (radical + THETA_PARAMETER) / sigmaSquared;
		final double expectedM = (radical - THETA_PARAMETER) / sigmaSquared;

		assertEquals(expectedC, jumpComponent.getC(), TOL);
		assertEquals(expectedG, jumpComponent.getG(), TOL);
		assertEquals(expectedM, jumpComponent.getM(), TOL);
	}

	@Test
	public void testCloneWithModifiedSpaceTimeDiscretization() {
		final SpaceTimeDiscretization originalDiscretization = createSpaceTimeDiscretization(10, 80);
		final SpaceTimeDiscretization newDiscretization = createSpaceTimeDiscretization(20, 120);

		final FDMVarianceGammaModel model = new FDMVarianceGammaModel(
				INITIAL_VALUE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_RATE,
				SIGMA,
				NU,
				THETA_PARAMETER,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				originalDiscretization
		);

		final FiniteDifferenceEquityModel clonedModel =
				model.getCloneWithModifiedSpaceTimeDiscretization(newDiscretization);

		assertTrue(clonedModel instanceof FDMVarianceGammaModel);

		final FDMVarianceGammaModel clonedVarianceGammaModel =
				(FDMVarianceGammaModel) clonedModel;

		assertArrayEquals(model.getInitialValue(), clonedVarianceGammaModel.getInitialValue(), TOL);

		final double time = 1.0;
		final double stateVariable = 110.0;

		assertEquals(
				model.getDrift(time, stateVariable)[0],
				clonedVarianceGammaModel.getDrift(time, stateVariable)[0],
				TOL
		);
		assertEquals(
				model.getFactorLoading(time, stateVariable)[0][0],
				clonedVarianceGammaModel.getFactorLoading(time, stateVariable)[0][0],
				TOL
		);

		assertTrue(clonedVarianceGammaModel.getJumpComponent().isPresent());
		assertTrue(clonedVarianceGammaModel.getJumpComponent().get() instanceof VarianceGammaJumpComponent);

		final VarianceGammaJumpComponent originalJumpComponent =
				model.getVarianceGammaJumpComponent();
		final VarianceGammaJumpComponent clonedJumpComponent =
				clonedVarianceGammaModel.getVarianceGammaJumpComponent();

		assertEquals(originalJumpComponent.getC(), clonedJumpComponent.getC(), TOL);
		assertEquals(originalJumpComponent.getG(), clonedJumpComponent.getG(), TOL);
		assertEquals(originalJumpComponent.getM(), clonedJumpComponent.getM(), TOL);
		assertEquals(
				originalJumpComponent.getLowerIntegrationBound(time, stateVariable),
				clonedJumpComponent.getLowerIntegrationBound(time, stateVariable),
				TOL
		);
		assertEquals(
				originalJumpComponent.getUpperIntegrationBound(time, stateVariable),
				clonedJumpComponent.getUpperIntegrationBound(time, stateVariable),
				TOL
		);

		assertSame(originalDiscretization, model.getSpaceTimeDiscretization());
		assertSame(newDiscretization, clonedVarianceGammaModel.getSpaceTimeDiscretization());
	}

	private SpaceTimeDiscretization createSpaceTimeDiscretization(
			final int numberOfTimeSteps,
			final int numberOfSpaceSteps) {

		final double maturity = 1.0;

		final double sMin = 40.0;
		final double sMax = 160.0;

		final Grid spaceGrid = new UniformGrid(numberOfSpaceSteps, sMin, sMax);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						numberOfTimeSteps,
						maturity / numberOfTimeSteps
				);

		return new SpaceTimeDiscretization(
				spaceGrid,
				timeDiscretization,
				THETA,
				new double[] { INITIAL_VALUE }
		);
	}
}