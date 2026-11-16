(function attachRetryPolicy(root, factory) {
    const policy = factory();
    if (typeof module === 'object' && module.exports) module.exports = policy;
    if (root) root.ClassMindRetryPolicy = policy;
})(typeof globalThis === 'object' ? globalThis : this, function createRetryPolicy() {
    function clampRandom(random) {
        const value = Number(random());
        return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0.5;
    }

    function parseRetryAfterMs(value, nowMs = Date.now()) {
        if (value === null || value === undefined) return null;
        const headerValue = String(value).trim();
        if (!headerValue) return null;

        const seconds = Number(headerValue);
        if (Number.isFinite(seconds) && seconds >= 0) {
            return Math.round(seconds * 1000);
        }

        const retryAtMs = Date.parse(headerValue);
        return Number.isFinite(retryAtMs) ? Math.max(0, retryAtMs - nowMs) : null;
    }

    function pollIntervalMs(isVisible, random = Math.random) {
        const baseMs = isVisible ? 5000 : 15000;
        return baseMs + Math.floor(clampRandom(random) * 1000);
    }

    function computeRetryDelayMs(attempt, retryAfterMs = null, random = Math.random) {
        const normalizedAttempt = Number.isFinite(Number(attempt))
            ? Math.max(1, Math.floor(Number(attempt)))
            : 1;
        const exponentialMs = Math.min(60000, 5000 * (2 ** (normalizedAttempt - 1)));
        const jitterMultiplier = 0.8 + clampRandom(random) * 0.4;
        const backoffMs = Math.round(exponentialMs * jitterMultiplier);
        const serverMinimumMs = Number.isFinite(retryAfterMs) && retryAfterMs >= 0
            ? retryAfterMs
            : 0;
        return Math.max(backoffMs, serverMinimumMs);
    }

    function shouldRetryPoll(status, attempt, maxAttempts = 5) {
        const normalizedStatus = Number(status);
        const isTransient = status === null || status === undefined ||
            normalizedStatus === 429 || normalizedStatus >= 500;
        const normalizedAttempt = Number(attempt);
        const normalizedLimit = Number(maxAttempts);
        return isTransient && Number.isInteger(normalizedAttempt) &&
            Number.isInteger(normalizedLimit) && normalizedAttempt >= 1 &&
            normalizedAttempt <= normalizedLimit;
    }

    function withJobPollLock(jobId, task, lockManager) {
        if (typeof task !== 'function') {
            throw new TypeError('A job polling task is required.');
        }
        if (!lockManager || typeof lockManager.request !== 'function') {
            return task();
        }
        return lockManager.request(
            `classmind-analysis-poll:${String(jobId)}`,
            { mode: 'exclusive' },
            task
        );
    }

    return Object.freeze({
        parseRetryAfterMs,
        pollIntervalMs,
        computeRetryDelayMs,
        shouldRetryPoll,
        withJobPollLock,
    });
});
