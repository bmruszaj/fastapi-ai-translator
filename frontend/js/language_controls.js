import {
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    SELECT_KEYS,
} from "./constants.js"
import { select_configs } from "./dom.js"
import { get_language_meta, normalize_language_code } from "./language_utils.js"

function create_native_option(language_code) {
    const option = document.createElement("option")
    const normalized_code = normalize_language_code(language_code)
    const meta = get_language_meta(normalized_code)

    option.value = normalized_code
    option.textContent = `${meta.name} (${normalized_code})`
    return option
}

function create_menu_option(select_key, language_code) {
    const normalized_code = normalize_language_code(language_code)
    const meta = get_language_meta(normalized_code)

    const list_item = document.createElement("li")
    const button = document.createElement("button")
    const flag = document.createElement("img")
    const label = document.createElement("span")

    list_item.setAttribute("role", "option")
    list_item.dataset.value = normalized_code

    button.type = "button"
    button.className = "select-option-button"
    button.dataset.value = normalized_code

    flag.className = "select-flag"
    flag.src = meta.flagPath
    flag.alt = ""

    label.textContent = `${meta.name} (${normalized_code})`

    button.append(flag, label)
    button.addEventListener("click", () => {
        set_select_value(select_key, normalized_code, true)
        close_all_selects()
    })

    list_item.appendChild(button)
    return list_item
}

export function update_select_trigger(select_key) {
    const config = select_configs[select_key]
    const selected_code = normalize_language_code(config.select.value)
    const meta = get_language_meta(selected_code)

    config.label.textContent = `${meta.name} (${selected_code})`
    config.flag.src = meta.flagPath
    config.flag.alt = ""
}

export function update_option_highlight(select_key) {
    const config = select_configs[select_key]
    const selected_code = normalize_language_code(config.select.value)
    const option_buttons = config.options.querySelectorAll(".select-option-button")

    for (const button of option_buttons) {
        const is_active = normalize_language_code(button.dataset.value) === selected_code
        button.classList.toggle("is-active", is_active)
        button.parentElement?.setAttribute("aria-selected", String(is_active))
    }
}

export function set_select_value(select_key, language_code, emit_change) {
    const config = select_configs[select_key]
    config.select.value = normalize_language_code(language_code)
    update_select_trigger(select_key)
    update_option_highlight(select_key)

    if (emit_change) {
        config.select.dispatchEvent(new Event("change", { bubbles: true }))
    }
}

export function populate_language_controls(languages) {
    for (const select_key of SELECT_KEYS) {
        const config = select_configs[select_key]
        config.select.innerHTML = ""
        config.options.innerHTML = ""

        for (const language_code of languages) {
            config.select.appendChild(create_native_option(language_code))
            config.options.appendChild(create_menu_option(select_key, language_code))
        }

        update_select_trigger(select_key)
        update_option_highlight(select_key)
    }
}

function resolve_default_language(preferred_code, available_languages, fallback_index) {
    if (available_languages.includes(preferred_code)) {
        return preferred_code
    }

    if (available_languages.length === 0) {
        return ""
    }

    return available_languages[Math.min(fallback_index, available_languages.length - 1)]
}

export function apply_default_language_pair(languages) {
    const source_language = resolve_default_language(DEFAULT_SOURCE_LANGUAGE, languages, 0)
    const target_language = resolve_default_language(DEFAULT_TARGET_LANGUAGE, languages, 1)

    set_select_value("source", source_language, false)
    if (source_language !== target_language) {
        set_select_value("target", target_language, false)
        return
    }

    const alternative_target =
        languages.find((language_code) => language_code !== source_language) || source_language
    set_select_value("target", alternative_target, false)
}

export function close_all_selects() {
    for (const select_key of SELECT_KEYS) {
        const config = select_configs[select_key]
        config.root.classList.remove("open")
        config.trigger.setAttribute("aria-expanded", "false")
    }
}

export function toggle_select(select_key) {
    const config = select_configs[select_key]
    const should_open = !config.root.classList.contains("open")

    close_all_selects()
    if (!should_open) {
        return
    }

    config.root.classList.add("open")
    config.trigger.setAttribute("aria-expanded", "true")
}
