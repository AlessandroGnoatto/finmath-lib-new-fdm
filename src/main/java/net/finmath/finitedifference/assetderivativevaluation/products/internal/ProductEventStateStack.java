package net.finmath.finitedifference.assetderivativevaluation.products.internal;

import java.util.ArrayDeque;

/**
 * Thread-local stack for temporary product valuation state.
 *
 * <p>
 * Intended for product-local event states used during nested or concurrent
 * finite-difference valuations.
 * </p>
 *
 * @param <T> State type.
 * @author Alessandro Gnoatto
 */
public final class ProductEventStateStack<T> {

	private transient ThreadLocal<ArrayDeque<T>> stack;

	public Scope push(final T state) {
		if(state == null) {
			throw new IllegalArgumentException("state must not be null.");
		}

		getStack().get().push(state);

		return this::pop;
	}

	public T currentOrNull() {
		final ArrayDeque<T> currentStack = getStack().get();
		return currentStack.isEmpty() ? null : currentStack.peek();
	}

	private void pop() {
		final ArrayDeque<T> currentStack = getStack().get();

		if(currentStack.isEmpty()) {
			throw new IllegalStateException("No product event state to pop.");
		}

		currentStack.pop();

		if(currentStack.isEmpty()) {
			getStack().remove();
		}
	}

	private ThreadLocal<ArrayDeque<T>> getStack() {
		if(stack == null) {
			stack = ThreadLocal.withInitial(ArrayDeque::new);
		}

		return stack;
	}

	@FunctionalInterface
	public interface Scope extends AutoCloseable {

		@Override
		void close();
	}
}