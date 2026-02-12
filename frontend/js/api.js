import { API_BASE } from "./constants.js"
import { enforce_required_languages, normalize_language_code } from "./language_utils.js"

function extract_api_error_message(response_body, fallback_message) {
    if (response_body && typeof response_body.message === "string" && response_body.message.trim()) {
        return response_body.message.trim()
    }

    if (response_body && typeof response_body.detail === "string" && response_body.detail.trim()) {
        return response_body.detail.trim()
    }

    return fallback_message
}

export async function fetch_frontend_config() {
    const response = await fetch(`${API_BASE}/frontend-config`, {
        headers: {
            Accept: "application/json",
        },
    })

    let response_body = {}
    try {
        response_body = await response.json()
    } catch (_error) {
        response_body = {}
    }

    if (!response.ok) {
        const error_message = extract_api_error_message(
            response_body,
            `Could not load frontend limits (HTTP ${response.status}).`
        )
        throw new Error(error_message)
    }

    return response_body
}

export async function fetch_languages() {
    const response = await fetch(`${API_BASE}/languages`, {
        headers: {
            Accept: "application/json",
        },
    })

    let response_body = {}
    try {
        response_body = await response.json()
    } catch (_error) {
        response_body = {}
    }

    if (!response.ok) {
        const error_message = extract_api_error_message(
            response_body,
            `Could not load supported languages (HTTP ${response.status}).`
        )
        throw new Error(error_message)
    }

    const raw_languages = Array.isArray(response_body.languages) ? response_body.languages : []
    const languages = raw_languages.map(normalize_language_code).filter((code) => code.length > 0)
    const enforced_languages = enforce_required_languages(languages)

    if (enforced_languages.length < 2) {
        throw new Error("At least two supported languages are required.")
    }

    return enforced_languages
}

export async function translate_text(payload) {
    const response = await fetch(`${API_BASE}/translate`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
        },
        body: JSON.stringify(payload),
    })

    let response_body = {}
    try {
        response_body = await response.json()
    } catch (_error) {
        response_body = {}
    }

    if (!response.ok) {
        const is_server_error = response.status >= 500
        const error_message = is_server_error
            ? "Translation failed. Please try again later."
            : extract_api_error_message(
                  response_body,
                  `Translation request is invalid (HTTP ${response.status}).`
              )
        throw new Error(error_message)
    }

    return response_body
}
