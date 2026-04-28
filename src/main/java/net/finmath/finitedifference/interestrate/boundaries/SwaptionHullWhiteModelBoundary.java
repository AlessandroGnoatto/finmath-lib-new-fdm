package net.finmath.finitedifference.interestrate.boundaries;

import java.util.ArrayList;
import java.util.List;

import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.boundaries.StandardBoundaryCondition;
import net.finmath.finitedifference.interestrate.models.FDMHullWhiteModel;
import net.finmath.finitedifference.interestrate.products.FiniteDifferenceInterestRateProduct;
import net.finmath.finitedifference.interestrate.products.Swaption;
import net.finmath.modelling.Exercise;
import net.finmath.montecarlo.interestrate.products.BermudanSwaptionFromSwapSchedules.SwaptionType;
import net.finmath.time.Schedule;
import net.finmath.time.TimeDiscretization;

/**
 * Boundary conditions for {@link Swaption} under {@link FDMHullWhiteModel}.
 *
 * <p>
 * This implementation provides asymptotic Dirichlet boundary values compatible
 * with European, Bermudan, and American swaption exercise.
 * </p>
 *
 * <p>
 * The guiding asymptotics are:
 * </p>
 * <ul>
 *   <li>on the deep out-of-the-money side, the swaption value tends to zero,</li>
 *   <li>on the deep in-the-money side, the swaption value approaches the
 *       intrinsic value envelope over the remaining admissible exercise
 *       opportunities.</li>
 * </ul>
 *
 * <p>
 * Concretely:
 * </p>
 * <ul>
 *   <li>for a {@code RECEIVER} swaption, the lower boundary is treated as the
 *       deep in-the-money side and the upper boundary as the deep
 *       out-of-the-money side,</li>
 *   <li>for a {@code PAYER} swaption, the upper boundary is treated as the deep
 *       in-the-money side and the lower boundary as the deep out-of-the-money
 *       side.</li>
 * </ul>
 *
 * <p>
 * On the deep in-the-money side, the boundary value is chosen as
 * </p>
 *
 * <p>
 * <i>
 * \max_{j \in \mathcal{E}(t)} \bigl( V_{\mathrm{swap},j}(t,x), 0 \bigr),
 * </i>
 * </p>
 *
 * <p>
 * where {@code \mathcal{E}(t)} denotes the set of remaining admissible exercise
 * opportunities and {@code V_swap,j(t,x)} is the value of the corresponding
 * underlying swap.
 * </p>
 *
 * <p>
 * This is a pragmatic boundary choice for the unified PDE swaption class.
 * A European-only Jamshidian-exact boundary can be added later as a refinement.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class SwaptionHullWhiteModelBoundary implements FiniteDifferenceInterestRateBoundary {

	private static final double TIME_TOLERANCE = 1E-12;

	private final FDMHullWhiteModel model;

	/**
	 * Creates the Hull-White boundary for {@link Swaption}.
	 *
	 * @param model The Hull-White model.
	 */
	public SwaptionHullWhiteModelBoundary(final FDMHullWhiteModel model) {
		if(model == null) {
			throw new IllegalArgumentException("model must not be null.");
		}
		if(model.getInitialValue() == null || model.getInitialValue().length != 1) {
			throw new IllegalArgumentException(
					"SwaptionHullWhiteModelBoundary requires a one-dimensional Hull-White model."
			);
		}

		this.model = model;
	}

	@Override
	public BoundaryCondition[] getBoundaryConditionsAtLowerBoundary(
			final FiniteDifferenceInterestRateProduct product,
			final double time,
			final double... stateVariables) {

		final Swaption swaption = validateAndCastProduct(product);
		validateStateVariables(stateVariables);

		final double boundaryValue = getLowerBoundaryValue(swaption, time, stateVariables[0]);

		return new BoundaryCondition[] {
				StandardBoundaryCondition.dirichlet(boundaryValue)
		};
	}

	@Override
	public BoundaryCondition[] getBoundaryConditionsAtUpperBoundary(
			final FiniteDifferenceInterestRateProduct product,
			final double time,
			final double... stateVariables) {

		final Swaption swaption = validateAndCastProduct(product);
		validateStateVariables(stateVariables);

		final double boundaryValue = getUpperBoundaryValue(swaption, time, stateVariables[0]);

		return new BoundaryCondition[] {
				StandardBoundaryCondition.dirichlet(boundaryValue)
		};
	}

	private Swaption validateAndCastProduct(final FiniteDifferenceInterestRateProduct product) {
		if(!(product instanceof Swaption)) {
			throw new IllegalArgumentException(
					"SwaptionHullWhiteModelBoundary requires a Swaption product."
			);
		}

		return (Swaption) product;
	}

	private void validateStateVariables(final double[] stateVariables) {
		if(stateVariables == null || stateVariables.length != 1) {
			throw new IllegalArgumentException("Exactly one state variable is required.");
		}
	}

	private double getLowerBoundaryValue(
			final Swaption swaption,
			final double time,
			final double stateVariable) {

		switch(swaption.getSwaptionType()) {
		case RECEIVER:
			return getDeepInTheMoneyBoundaryValue(swaption, time, stateVariable);
		case PAYER:
			return 0.0;
		default:
			throw new IllegalArgumentException("Unsupported swaption type: " + swaption.getSwaptionType());
		}
	}

	private double getUpperBoundaryValue(
			final Swaption swaption,
			final double time,
			final double stateVariable) {

		switch(swaption.getSwaptionType()) {
		case RECEIVER:
			return 0.0;
		case PAYER:
			return getDeepInTheMoneyBoundaryValue(swaption, time, stateVariable);
		default:
			throw new IllegalArgumentException("Unsupported swaption type: " + swaption.getSwaptionType());
		}
	}

	private double getDeepInTheMoneyBoundaryValue(
			final Swaption swaption,
			final double time,
			final double stateVariable) {

		final List<ExerciseOpportunity> remainingExerciseOpportunities =
				getRemainingExerciseOpportunities(swaption, time);

		if(remainingExerciseOpportunities.isEmpty()) {
			return 0.0;
		}

		double value = 0.0;

		for(final ExerciseOpportunity opportunity : remainingExerciseOpportunities) {
			final double swapValue = getUnderlyingSwapValue(
					swaption,
					time,
					opportunity.scheduleIndex,
					stateVariable
			);

			value = Math.max(value, Math.max(swapValue, 0.0));
		}

		return value;
	}

	private List<ExerciseOpportunity> getRemainingExerciseOpportunities(
			final Swaption swaption,
			final double time) {

		final Exercise exercise = swaption.getExercise();
		final List<ExerciseOpportunity> opportunities = new ArrayList<>();

		if(exercise.isAmerican()) {
			if(time <= exercise.getMaturity() + TIME_TOLERANCE) {
				opportunities.add(new ExerciseOpportunity(time, 0));
			}
			return opportunities;
		}

		final double[] exerciseTimes = resolveDiscreteExerciseTimes(swaption);
		final int[] scheduleIndices = resolveDiscreteScheduleIndices(swaption, exerciseTimes);

		for(int i = 0; i < exerciseTimes.length; i++) {
			if(exerciseTimes[i] >= time - TIME_TOLERANCE) {
				opportunities.add(new ExerciseOpportunity(exerciseTimes[i], scheduleIndices[i]));
			}
		}

		return opportunities;
	}

	private double[] resolveDiscreteExerciseTimes(final Swaption swaption) {

		final double[] explicitExerciseDates = swaption.getExplicitExerciseDates();
		if(explicitExerciseDates != null) {
			return explicitExerciseDates.clone();
		}

		final Exercise exercise = swaption.getExercise();

		if(exercise.isEuropean()) {
			return new double[] { exercise.getMaturity() };
		}

		if(exercise.isBermudan()) {
			final Schedule[] fixSchedules = swaption.getFixSchedules();

			if(fixSchedules.length == 1) {
				return new double[] { exercise.getMaturity() };
			}

			final double[] exerciseTimes = new double[fixSchedules.length];
			for(int i = 0; i < exerciseTimes.length; i++) {
				exerciseTimes[i] = inferExerciseDateFromSchedules(
						swaption.getFixSchedules()[i],
						swaption.getFloatSchedules()[i],
						i
				);
			}
			return exerciseTimes;
		}

		throw new IllegalArgumentException("Unsupported exercise type.");
	}

	private int[] resolveDiscreteScheduleIndices(
			final Swaption swaption,
			final double[] exerciseTimes) {

		final Schedule[] fixSchedules = swaption.getFixSchedules();

		if(fixSchedules.length == 1) {
			final int[] scheduleIndices = new int[exerciseTimes.length];
			for(int i = 0; i < scheduleIndices.length; i++) {
				scheduleIndices[i] = 0;
			}
			return scheduleIndices;
		}

		if(fixSchedules.length != exerciseTimes.length) {
			throw new IllegalArgumentException(
					"The number of underlying swap definitions must be either 1 or equal to the number of exercise dates."
			);
		}

		final int[] scheduleIndices = new int[exerciseTimes.length];
		for(int i = 0; i < scheduleIndices.length; i++) {
			scheduleIndices[i] = i;
		}

		return scheduleIndices;
	}

	private double inferExerciseDateFromSchedules(
			final Schedule fixSchedule,
			final Schedule floatSchedule,
			final int scheduleIndex) {

		final double fixStart = fixSchedule.getPeriodStart(0);
		final double floatStart = floatSchedule.getPeriodStart(0);

		if(Math.abs(fixStart - floatStart) > TIME_TOLERANCE) {
			throw new IllegalArgumentException(
					"Cannot infer a unique exercise date from schedule pair "
					+ scheduleIndex
					+ ": first fixed-leg start and first float-leg start differ."
			);
		}

		return fixStart;
	}

	private double getUnderlyingSwapValue(
			final Swaption swaption,
			final double time,
			final int scheduleIndex,
			final double stateVariable) {

		final double fixedLegValue = getFixedLegValue(
				swaption,
				time,
				scheduleIndex,
				stateVariable
		);

		final double floatingLegValue = getFloatingLegValue(
				swaption,
				time,
				scheduleIndex,
				stateVariable
		);

		if(swaption.getSwaptionType() == SwaptionType.PAYER) {
			return floatingLegValue - fixedLegValue;
		}
		else if(swaption.getSwaptionType() == SwaptionType.RECEIVER) {
			return fixedLegValue - floatingLegValue;
		}
		else {
			throw new IllegalArgumentException("Unsupported swaption type: " + swaption.getSwaptionType());
		}
	}

	private double getFixedLegValue(
			final Swaption swaption,
			final double time,
			final int scheduleIndex,
			final double stateVariable) {

		final Schedule fixedSchedule = swaption.getFixSchedules()[scheduleIndex];
		final double fixedRate = swaption.getSwapRates()[scheduleIndex];
		final double notional = swaption.getNotionals()[scheduleIndex];

		double value = 0.0;

		for(int periodIndex = 0; periodIndex < fixedSchedule.getNumberOfPeriods(); periodIndex++) {
			final double paymentTime = fixedSchedule.getPayment(periodIndex);

			if(paymentTime <= time + TIME_TOLERANCE) {
				continue;
			}

			final double accrualFactor = fixedSchedule.getPeriodLength(periodIndex);
			final double discountBond = model.getDiscountBond(time, paymentTime, stateVariable);

			value += notional * fixedRate * accrualFactor * discountBond;
		}

		return value;
	}

	private double getFloatingLegValue(
			final Swaption swaption,
			final double time,
			final int scheduleIndex,
			final double stateVariable) {

		final Schedule floatingSchedule = swaption.getFloatSchedules()[scheduleIndex];
		final double notional = swaption.getNotionals()[scheduleIndex];

		double value = 0.0;

		for(int periodIndex = 0; periodIndex < floatingSchedule.getNumberOfPeriods(); periodIndex++) {
			final double fixingTime = floatingSchedule.getFixing(periodIndex);
			final double paymentTime = floatingSchedule.getPayment(periodIndex);

			if(fixingTime < time - TIME_TOLERANCE) {
				continue;
			}
			if(paymentTime <= time + TIME_TOLERANCE) {
				continue;
			}

			final double accrualFactor = floatingSchedule.getPeriodLength(periodIndex);
			final double forwardRate = model.getForwardRate(
					swaption.getForwardCurveName(),
					time,
					fixingTime,
					paymentTime,
					stateVariable
			);
			final double discountBond = model.getDiscountBond(
					time,
					paymentTime,
					stateVariable
			);

			value += notional * forwardRate * accrualFactor * discountBond;
		}

		return value;
	}

	/**
	 * Small container for one exercise opportunity.
	 *
	 * @author Alessandro Gnoatto
	 */
	private static final class ExerciseOpportunity {

		private final double exerciseTime;
		private final int scheduleIndex;

		private ExerciseOpportunity(final double exerciseTime, final int scheduleIndex) {
			this.exerciseTime = exerciseTime;
			this.scheduleIndex = scheduleIndex;
		}
	}
}