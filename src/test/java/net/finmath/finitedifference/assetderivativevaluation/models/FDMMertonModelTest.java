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
 * Unit tests for {@link FDMMertonModel}.
 *
 * <p>
 * These tests validate:
 * </p>
 * <ul>
 *   <li>initial value and local coefficient contract,</li>
 *   <li>presence and wiring of the jump component,</li>
 *   <li>correct cloning with modified space-time discretization.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class FDMMertonModelTest {

	private static final double INITIAL_VALUE = 100.0;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD_RATE = 0.02;
	private static final double VOLATILITY = 0.30;

	private static final double JUMP_INTENSITY = 1.4;
	private static final double JUMP_MEAN = -0.10;
	private static final double JUMP_STD_DEV = 0.25;
	private static final double LOWER_INTEGRATION_BOUND = -2.0;
	private static final double UPPER_INTEGRATION_BOUND = 2.0;

	private static final double THETA = 0.5;

	private static final double TOL = 1E-12;

	@Test
	public void testInitialValueAndLocalCoefficients() {
		final SpaceTimeDiscretization spaceTimeDiscretization = createSpaceTimeDiscretization(10, 80);

		final FDMMertonModel model = new FDMMertonModel(
				INITIAL_VALUE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_RATE,
				VOLATILITY,
				JUMP_INTENSITY,
				JUMP_MEAN,
				JUMP_STD_DEV,
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

		assertEquals(expectedDrift, drift[0], 1E-12);
		final double[][] factorLoading = model.getFactorLoading(time, stateVariable);
		assertEquals(1, factorLoading.length);
		assertEquals(1, factorLoading[0].length);
		assertEquals(VOLATILITY * stateVariable, factorLoading[0][0], TOL);
	}

	@Test
	public void testJumpComponentPresenceAndParameters() {
		final SpaceTimeDiscretization spaceTimeDiscretization = createSpaceTimeDiscretization(10, 80);

		final FDMMertonModel model = new FDMMertonModel(
				INITIAL_VALUE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_RATE,
				VOLATILITY,
				JUMP_INTENSITY,
				JUMP_MEAN,
				JUMP_STD_DEV,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				spaceTimeDiscretization
		);

		final Optional<JumpComponent> optionalJumpComponent = model.getJumpComponent();

		assertTrue(optionalJumpComponent.isPresent());
		assertTrue(optionalJumpComponent.get() instanceof MertonJumpComponent);

		final MertonJumpComponent jumpComponent = (MertonJumpComponent) optionalJumpComponent.get();

		assertEquals(JUMP_INTENSITY, jumpComponent.getIntensity(), TOL);
		assertEquals(JUMP_MEAN, jumpComponent.getJumpMean(), TOL);
		assertEquals(JUMP_STD_DEV, jumpComponent.getJumpStdDev(), TOL);
		assertEquals(LOWER_INTEGRATION_BOUND, jumpComponent.getLowerIntegrationBound(0.5, INITIAL_VALUE), TOL);
		assertEquals(UPPER_INTEGRATION_BOUND, jumpComponent.getUpperIntegrationBound(0.5, INITIAL_VALUE), TOL);
	}

	@Test
	public void testCloneWithModifiedSpaceTimeDiscretization() {
		final SpaceTimeDiscretization originalDiscretization = createSpaceTimeDiscretization(10, 80);
		final SpaceTimeDiscretization newDiscretization = createSpaceTimeDiscretization(20, 120);

		final FDMMertonModel model = new FDMMertonModel(
				INITIAL_VALUE,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_RATE,
				VOLATILITY,
				JUMP_INTENSITY,
				JUMP_MEAN,
				JUMP_STD_DEV,
				LOWER_INTEGRATION_BOUND,
				UPPER_INTEGRATION_BOUND,
				originalDiscretization
		);

		final FiniteDifferenceEquityModel clonedModel =
				model.getCloneWithModifiedSpaceTimeDiscretization(newDiscretization);

		assertTrue(clonedModel instanceof FDMMertonModel);

		final FDMMertonModel clonedMertonModel = (FDMMertonModel) clonedModel;

		assertArrayEquals(model.getInitialValue(), clonedMertonModel.getInitialValue(), TOL);
		assertEquals(model.getVolatility(), clonedMertonModel.getVolatility(), TOL);

		final double time = 1.0;
		final double stateVariable = 110.0;

		assertEquals(
				model.getDrift(time, stateVariable)[0],
				clonedMertonModel.getDrift(time, stateVariable)[0],
				1E-10
		);
		assertEquals(
				model.getFactorLoading(time, stateVariable)[0][0],
				clonedMertonModel.getFactorLoading(time, stateVariable)[0][0],
				TOL
		);

		assertTrue(clonedMertonModel.getJumpComponent().isPresent());
		assertTrue(clonedMertonModel.getJumpComponent().get() instanceof MertonJumpComponent);

		final MertonJumpComponent originalJumpComponent = model.getMertonJumpComponent();
		final MertonJumpComponent clonedJumpComponent = clonedMertonModel.getMertonJumpComponent();

		assertEquals(originalJumpComponent.getIntensity(), clonedJumpComponent.getIntensity(), TOL);
		assertEquals(originalJumpComponent.getJumpMean(), clonedJumpComponent.getJumpMean(), TOL);
		assertEquals(originalJumpComponent.getJumpStdDev(), clonedJumpComponent.getJumpStdDev(), TOL);
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
		assertSame(newDiscretization, clonedMertonModel.getSpaceTimeDiscretization());
	}

	private SpaceTimeDiscretization createSpaceTimeDiscretization(
			final int numberOfTimeSteps,
			final int numberOfSpaceSteps) {

		final double maturity = 1.0;
		final double standardDeviation = VOLATILITY * INITIAL_VALUE * Math.sqrt(maturity);

		final double sMin = Math.max(0.0, INITIAL_VALUE - 6.0 * standardDeviation);
		final double sMax = INITIAL_VALUE + 6.0 * standardDeviation;

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