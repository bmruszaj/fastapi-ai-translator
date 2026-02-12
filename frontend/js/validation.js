import { elements } from "./dom.js"
import { normalize_language_code } from "./language_utils.js"
import { get_frontend_limits } from "./ui.js"

export function build_cache_key(text, source_language, target_language) {
    return JSON.stringify([source_language, target_language, text])
}

export function validate_form() {
    const frontend_limits = get_frontend_limits()
    const text = elements.sourceText.value.trim()
    if (!text) {
        return { valid: false, message: "Source text must not be empty." }
    }
    if (text.length > frontend_limits.max_input_chars) {
        return {
            valid: false,
            message: `Source text exceeds ${frontend_limits.max_input_chars} characters.`,
        }
    }

    const source_language = normalize_language_code(elements.sourceLanguage.value)
    const target_language = normalize_language_code(elements.targetLanguage.value)
    if (source_language === target_language) {
        return { valid: false, message: "Source and target language must be different." }
    }

    return {
        valid: true,
        text,
        source_language,
        target_language,
    }
}
