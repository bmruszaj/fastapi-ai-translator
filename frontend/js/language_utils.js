import { LANGUAGE_META, REQUIRED_LANGUAGES, UNKNOWN_LANGUAGE } from "./constants.js"

export function normalize_language_code(value) {
    return String(value || "").trim().toLowerCase()
}

export function get_language_meta(language_code) {
    const normalized_code = normalize_language_code(language_code)
    return LANGUAGE_META[normalized_code] || {
        name: normalized_code || UNKNOWN_LANGUAGE.name,
        flagPath: UNKNOWN_LANGUAGE.flagPath,
    }
}

export function enforce_required_languages(languages) {
    const normalized_languages = [...new Set(languages.map(normalize_language_code))]
    const supported_languages = REQUIRED_LANGUAGES.filter((code) =>
        normalized_languages.includes(code)
    )
    const missing_languages = REQUIRED_LANGUAGES.filter((code) => !supported_languages.includes(code))

    if (missing_languages.length > 0) {
        throw new Error(
            `Language list is incomplete. Missing: ${missing_languages.join(", ")}.`
        )
    }

    return supported_languages
}
