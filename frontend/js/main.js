import { fetch_frontend_config, fetch_languages, translate_text } from "./api.js"
import { elements } from "./dom.js"
import {
    apply_default_language_pair,
    close_all_selects,
    populate_language_controls,
    set_select_value,
    toggle_select,
    update_option_highlight,
    update_select_trigger,
} from "./language_controls.js"
import {
    apply_frontend_limits,
    clear_feedback,
    get_frontend_limits,
    reset_copy_button_feedback,
    set_copy_button_feedback,
    set_error,
    set_loading_state,
    set_status,
    set_translated_text,
    show_copy_success_feedback,
    update_character_feedback,
} from "./ui.js"
import { build_cache_key, validate_form } from "./validation.js"

const translation_cache = new Map()

async function handle_form_submit(event) {
    event.preventDefault()
    clear_feedback()

    const validation = validate_form()
    if (!validation.valid) {
        set_error(validation.message)
        return
    }

    const cache_key = build_cache_key(
        validation.text,
        validation.source_language,
        validation.target_language
    )
    if (translation_cache.has(cache_key)) {
        const cached_translation = translation_cache.get(cache_key) || ""
        set_translated_text(cached_translation)
        set_status(
            `Loaded from cache: ${validation.source_language.toUpperCase()} -> ${validation.target_language.toUpperCase()}.`
        )
        return
    }

    set_loading_state(true)
    set_status("Translating...", "loading")

    try {
        const response_body = await translate_text({
            text: validation.text,
            source_language: validation.source_language,
            target_language: validation.target_language,
        })
        const translated_text = response_body.translated_text || ""
        set_translated_text(translated_text)
        translation_cache.set(cache_key, translated_text)
        set_status(
            `Translated ${validation.source_language.toUpperCase()} -> ${validation.target_language.toUpperCase()}.`
        )
    } catch (error) {
        const fallback_message = "Translation failed. Please try again later."
        set_error(error instanceof Error ? error.message : fallback_message)
        set_translated_text("")
    } finally {
        set_loading_state(false)
    }
}

function handle_swap_languages() {
    clear_feedback()
    const source_language = elements.sourceLanguage.value
    const target_language = elements.targetLanguage.value
    const translated_text = elements.translatedText.value.trim()

    set_select_value("source", target_language, true)
    set_select_value("target", source_language, true)
    if (translated_text) {
        elements.sourceText.value = translated_text
        update_character_feedback()
    }
    set_translated_text("")
    set_status("Languages swapped.")
}

async function handle_copy_result() {
    const text = elements.translatedText.value
    if (!text.trim()) {
        return
    }

    try {
        await navigator.clipboard.writeText(text)
        show_copy_success_feedback()
        set_status("Translated text copied to clipboard.")
        set_error("")
    } catch (_error) {
        set_copy_button_feedback(false)
        set_error("Clipboard access failed. Copy manually from the output box.")
    }
}

function attach_event_listeners() {
    elements.form.addEventListener("submit", handle_form_submit)
    elements.swapButton.addEventListener("click", handle_swap_languages)
    elements.copyButton.addEventListener("click", handle_copy_result)

    elements.sourceLanguageButton.addEventListener("click", () => toggle_select("source"))
    elements.targetLanguageButton.addEventListener("click", () => toggle_select("target"))

    document.addEventListener("click", (event) => {
        const click_target = event.target
        if (!(click_target instanceof Node)) {
            return
        }

        const inside_source = elements.sourceSelectRoot.contains(click_target)
        const inside_target = elements.targetSelectRoot.contains(click_target)
        if (!inside_source && !inside_target) {
            close_all_selects()
        }
    })

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            close_all_selects()
        }
    })

    elements.sourceText.addEventListener("input", () => {
        set_error("")
        update_character_feedback()
    })

    elements.sourceLanguage.addEventListener("change", () => {
        set_error("")
        update_select_trigger("source")
        update_option_highlight("source")
    })

    elements.targetLanguage.addEventListener("change", () => {
        set_error("")
        update_select_trigger("target")
        update_option_highlight("target")
    })
}

async function initialize_page() {
    let initialization_failed = false
    set_loading_state(true)
    clear_feedback()
    reset_copy_button_feedback()
    set_translated_text("")
    elements.sourceText.maxLength = get_frontend_limits().max_input_chars
    update_character_feedback()
    attach_event_listeners()

    try {
        const frontend_config = await fetch_frontend_config()
        apply_frontend_limits(frontend_config)
        const languages = await fetch_languages()
        populate_language_controls(languages)
        apply_default_language_pair(languages)
        set_status("Ready to translate.")
    } catch (error) {
        initialization_failed = true
        const fallback_message = "Unable to initialize application."
        set_error(error instanceof Error ? error.message : fallback_message)
    } finally {
        set_loading_state(false)
        if (initialization_failed) {
            elements.translateButton.disabled = true
            elements.swapButton.disabled = true
            elements.sourceLanguageButton.disabled = true
            elements.targetLanguageButton.disabled = true
        }
    }
}

void initialize_page()
