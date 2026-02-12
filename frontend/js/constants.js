export const API_BASE = "/api"

export const COPY_BUTTON_DEFAULT_LABEL = "Copy result"
export const COPY_BUTTON_SUCCESS_LABEL = "Copied"

export const DEFAULT_SOURCE_LANGUAGE = "pl"
export const DEFAULT_TARGET_LANGUAGE = "en"

export const REQUIRED_LANGUAGES = ["de", "en", "el", "es", "fr", "it", "pl", "pt", "ro", "nl"]
export const SELECT_KEYS = ["source", "target"]

export const DEFAULT_MAX_INPUT_TOKENS = 400
export const DEFAULT_MAX_CHARS_PER_TOKEN = 2
export const DEFAULT_WARNING_RATIO = 0.75

export const UNKNOWN_LANGUAGE = {
    name: "Unknown",
    flagPath: "/assets/flags/unknown.svg",
}

export const LANGUAGE_META = {
    de: { name: "German", flagPath: "/assets/flags/de.svg" },
    en: { name: "English", flagPath: "/assets/flags/en.svg" },
    el: { name: "Greek", flagPath: "/assets/flags/el.svg" },
    es: { name: "Spanish", flagPath: "/assets/flags/es.svg" },
    fr: { name: "French", flagPath: "/assets/flags/fr.svg" },
    it: { name: "Italian", flagPath: "/assets/flags/it.svg" },
    pl: { name: "Polish", flagPath: "/assets/flags/pl.svg" },
    pt: { name: "Portuguese", flagPath: "/assets/flags/pt.svg" },
    ro: { name: "Romanian", flagPath: "/assets/flags/ro.svg" },
    nl: { name: "Dutch", flagPath: "/assets/flags/nl.svg" },
}
