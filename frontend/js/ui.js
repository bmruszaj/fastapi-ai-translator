import {
    COPY_BUTTON_DEFAULT_LABEL,
    COPY_BUTTON_SUCCESS_LABEL,
    DEFAULT_MAX_CHARS_PER_TOKEN,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_WARNING_RATIO,
} from "./constants.js"
import { elements, select_configs } from "./dom.js"

let is_loading = false
let copy_feedback_timeout_id = null
const frontend_limits = {
    max_input_tokens: DEFAULT_MAX_INPUT_TOKENS,
    max_chars_per_token: DEFAULT_MAX_CHARS_PER_TOKEN,
    max_input_chars: Math.floor(DEFAULT_MAX_INPUT_TOKENS * DEFAULT_MAX_CHARS_PER_TOKEN),
    warning_input_chars: Math.floor(
        DEFAULT_MAX_INPUT_TOKENS * DEFAULT_MAX_CHARS_PER_TOKEN * DEFAULT_WARNING_RATIO
    ),
}

export function set_loading_state(loading) {
    is_loading = loading
    elements.translateButton.disabled = loading
    elements.swapButton.disabled = loading
    elements.sourceText.disabled = loading
    select_configs.source.trigger.disabled = loading
    select_configs.target.trigger.disabled = loading
    elements.copyButton.disabled = loading || !elements.translatedText.value.trim()
    elements.translateButtonSpinner.hidden = !loading
    elements.translateButtonLabel.textContent = loading ? "Translating..." : "Translate"
}

export function set_copy_button_feedback(copied) {
    elements.copyButton.setAttribute(
        "aria-label",
        copied ? COPY_BUTTON_SUCCESS_LABEL : COPY_BUTTON_DEFAULT_LABEL
    )
    elements.copyButton.title = copied ? COPY_BUTTON_SUCCESS_LABEL : COPY_BUTTON_DEFAULT_LABEL
    elements.copyIconDefault.classList.toggle("is-visible", !copied)
    elements.copyIconSuccess.classList.toggle("is-visible", copied)
}

export function clear_copy_feedback_timer() {
    if (copy_feedback_timeout_id === null) {
        return
    }

    clearTimeout(copy_feedback_timeout_id)
    copy_feedback_timeout_id = null
}

export function reset_copy_button_feedback() {
    clear_copy_feedback_timer()
    set_copy_button_feedback(false)
}

export function show_copy_success_feedback() {
    clear_copy_feedback_timer()
    set_copy_button_feedback(true)
    copy_feedback_timeout_id = window.setTimeout(() => {
        set_copy_button_feedback(false)
        clear_copy_feedback_timer()
    }, 2000)
}

export function set_status(message, variant = "success") {
    if (!message) {
        elements.statusMessage.hidden = true
        elements.statusMessage.textContent = ""
        elements.statusMessage.classList.remove("is-loading")
        return
    }

    elements.statusMessage.classList.toggle("is-loading", variant === "loading")
    elements.statusMessage.textContent = message
    elements.statusMessage.hidden = false
}

export function set_error(message) {
    if (!message) {
        elements.errorMessage.hidden = true
        elements.errorMessage.textContent = ""
        return
    }

    const normalized_message = message.startsWith("Error:") ? message : `Error: ${message}`
    elements.errorMessage.textContent = normalized_message
    elements.errorMessage.hidden = false
    set_status("")
}

export function clear_feedback() {
    set_status("")
    set_error("")
}

export function update_character_feedback() {
    const text = elements.sourceText.value
    const character_count = text.length
    const is_limit_reached = character_count >= frontend_limits.max_input_chars
    const is_warning_threshold_reached =
        !is_limit_reached && character_count >= frontend_limits.warning_input_chars

    elements.charCounter.textContent = `${character_count} / ${frontend_limits.max_input_chars} chars`
    elements.charCounter.classList.toggle("is-warning", is_warning_threshold_reached)
    elements.charCounter.classList.toggle("is-limit", is_limit_reached)
    elements.sourceText.classList.toggle("is-warning-state", is_warning_threshold_reached)
    elements.sourceText.classList.toggle("is-limit-state", is_limit_reached)
    elements.lengthWarning.classList.toggle("is-limit", is_limit_reached)

    if (is_limit_reached) {
        elements.lengthWarning.hidden = false
        elements.lengthWarning.textContent = "Character limit reached."
        return
    }

    if (is_warning_threshold_reached) {
        const remaining_chars = frontend_limits.max_input_chars - character_count
        elements.lengthWarning.hidden = false
        elements.lengthWarning.textContent = `${remaining_chars} chars left.`
        return
    }

    elements.lengthWarning.hidden = true
    elements.lengthWarning.textContent = ""
}

export function set_translated_text(value) {
    elements.translatedText.value = value
    elements.copyButton.disabled = !value.trim() || is_loading
    if (!value.trim()) {
        reset_copy_button_feedback()
    }
}

function normalize_positive_integer(value) {
    const parsed_value = Number.parseInt(String(value), 10)
    if (!Number.isFinite(parsed_value) || parsed_value <= 0) {
        throw new Error("Frontend limit configuration is invalid.")
    }
    return parsed_value
}

export function apply_frontend_limits(limits) {
    frontend_limits.max_input_tokens = normalize_positive_integer(limits.max_input_tokens)
    frontend_limits.max_chars_per_token = normalize_positive_integer(
        limits.max_chars_per_token
    )
    frontend_limits.max_input_chars = normalize_positive_integer(limits.max_input_chars)
    frontend_limits.warning_input_chars = normalize_positive_integer(
        limits.warning_input_chars
    )

    if (frontend_limits.warning_input_chars >= frontend_limits.max_input_chars) {
        frontend_limits.warning_input_chars = frontend_limits.max_input_chars - 1
    }
    elements.sourceText.maxLength = frontend_limits.max_input_chars
    update_character_feedback()
}

export function get_frontend_limits() {
    return {
        max_input_tokens: frontend_limits.max_input_tokens,
        max_chars_per_token: frontend_limits.max_chars_per_token,
        max_input_chars: frontend_limits.max_input_chars,
        warning_input_chars: frontend_limits.warning_input_chars,
    }
}
