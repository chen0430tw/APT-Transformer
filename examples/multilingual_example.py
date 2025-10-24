#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Multilingual Support Example

Demonstrates the multilingual support system in APT.

Features demonstrated:
1. Language registry and lookup
2. Language detection
3. Script detection
4. Language features
5. Vocabulary size recommendations
"""

def demo_language_registry():
    """Demonstrate language registry functionality."""
    print("=" * 60)
    print("1. Language Registry")
    print("=" * 60)

    from apt.multilingual import (
        get_language,
        list_languages,
        get_vocab_size,
        is_language_supported
    )

    # Get a language by code
    print("\nGet language by code:")
    chinese = get_language('zh')
    print(f"  Code: {chinese.code}")
    print(f"  Name: {chinese.name}")
    print(f"  Native: {chinese.native_name}")
    print(f"  Script: {chinese.script.value}")
    print(f"  Vocab size: {chinese.vocab_size:,}")

    # Get language by name
    print("\nGet language by name:")
    japanese = get_language('Japanese')
    print(f"  {japanese.code}: {japanese.name} ({japanese.native_name})")

    # Get language by alias
    print("\nGet language by alias:")
    chinese2 = get_language('zh-cn')
    print(f"  Alias 'zh-cn' resolves to: {chinese2.name}")

    # List all languages
    print("\nAll supported languages:")
    for lang in list_languages():
        print(f"  {lang.code:6s} | {lang.name:25s} | {lang.native_name}")

    # Check if language is supported
    print("\nLanguage support check:")
    print(f"  Chinese supported: {is_language_supported('zh')}")
    print(f"  Klingon supported: {is_language_supported('tlh')}")

    # Get recommended vocab size
    print("\nRecommended vocabulary sizes:")
    for code in ['en', 'zh', 'ja', 'ko', 'ar']:
        vocab_size = get_vocab_size(code)
        lang = get_language(code)
        print(f"  {lang.name:20s}: {vocab_size:,}")


def demo_language_detection():
    """Demonstrate language detection."""
    print("\n" + "=" * 60)
    print("2. Language Detection")
    print("=" * 60)

    from apt.multilingual import detect_language, detect_script

    # Test texts in different languages
    test_texts = {
        'English': "Hello, world! How are you?",
        'Chinese': "你好世界！今天天气怎么样？",
        'Japanese': "こんにちは、世界！元気ですか？",
        'Korean': "안녕하세요 세계! 어떻게 지내세요?",
        'Arabic': "مرحبا بالعالم! كيف حالك؟",
        'Russian': "Привет мир! Как дела?",
        'Spanish': "¡Hola mundo! ¿Cómo estás?",
    }

    print("\nAutomatic language detection:")
    for expected, text in test_texts.items():
        detected = detect_language(text)
        script = detect_script(text)
        print(f"  {expected:10s} | Detected: {detected:6s} | Script: {script.value}")

    # Mixed language detection
    print("\nMixed language text:")
    from apt.multilingual import is_mixed_language, language_detector

    mixed_text = "Hello 你好 world 世界"
    is_mixed = is_mixed_language(mixed_text)
    dist = language_detector.get_script_distribution(mixed_text)

    print(f"  Text: {mixed_text}")
    print(f"  Is mixed: {is_mixed}")
    print(f"  Script distribution:")
    for script, proportion in dist.items():
        print(f"    {script}: {proportion:.1%}")


def demo_language_features():
    """Demonstrate language features."""
    print("\n" + "=" * 60)
    print("3. Language Features")
    print("=" * 60)

    from apt.multilingual import get_language, LanguageFeatures

    # Check features for different languages
    print("\nLanguage features:")

    # English
    english = get_language('en')
    print(f"\n  {english.name}:")
    print(f"    Has spaces: {english.has_feature(LanguageFeatures.SPACES)}")
    print(f"    Has tones: {english.has_feature(LanguageFeatures.TONES)}")
    print(f"    Has articles: {english.has_feature(LanguageFeatures.ARTICLES)}")
    print(f"    Word order: ", end="")
    if english.has_feature(LanguageFeatures.SVO):
        print("SVO (Subject-Verb-Object)")

    # Chinese
    chinese = get_language('zh')
    print(f"\n  {chinese.name}:")
    print(f"    Has spaces: {chinese.has_feature(LanguageFeatures.SPACES)}")
    print(f"    Has tones: {chinese.has_feature(LanguageFeatures.TONES)}")
    print(f"    Logographic: {chinese.has_feature(LanguageFeatures.LOGOGRAPHIC)}")
    print(f"    Needs segmentation: {chinese.has_feature(LanguageFeatures.NEEDS_SEGMENTATION)}")

    # Japanese
    japanese = get_language('ja')
    print(f"\n  {japanese.name}:")
    print(f"    Has spaces: {japanese.has_feature(LanguageFeatures.SPACES)}")
    print(f"    Mixed script: {japanese.has_feature(LanguageFeatures.MIXED_SCRIPT)}")
    print(f"    Syllabic: {japanese.has_feature(LanguageFeatures.SYLLABIC)}")
    print(f"    Word order: ", end="")
    if japanese.has_feature(LanguageFeatures.SOV):
        print("SOV (Subject-Object-Verb)")

    # Find all tonal languages
    print("\nTonal languages:")
    from apt.multilingual import language_registry
    tonal_langs = language_registry.get_languages_by_feature(LanguageFeatures.TONES)
    for lang in tonal_langs:
        print(f"  - {lang.name} ({lang.native_name})")


def demo_rtl_languages():
    """Demonstrate RTL (right-to-left) languages."""
    print("\n" + "=" * 60)
    print("4. Right-to-Left Languages")
    print("=" * 60)

    from apt.multilingual import language_registry, Direction

    rtl_langs = language_registry.get_rtl_languages()

    print(f"\nFound {len(rtl_langs)} RTL language(s):")
    for lang in rtl_langs:
        print(f"  {lang.name:15s} ({lang.native_name})")
        print(f"    Code: {lang.code}")
        print(f"    Script: {lang.script.value}")
        print(f"    Direction: {lang.direction.value}")


def demo_language_groups():
    """Demonstrate language groups."""
    print("\n" + "=" * 60)
    print("5. Language Groups")
    print("=" * 60)

    from apt.multilingual import (
        EAST_ASIAN_LANGUAGES,
        EUROPEAN_LANGUAGES,
        RTL_LANGUAGES
    )

    print("\nEast Asian languages:")
    for lang in EAST_ASIAN_LANGUAGES:
        print(f"  - {lang.name} ({lang.native_name})")

    print("\nEuropean languages:")
    for lang in EUROPEAN_LANGUAGES:
        print(f"  - {lang.name} ({lang.native_name})")

    print("\nRTL languages:")
    for lang in RTL_LANGUAGES:
        print(f"  - {lang.name} ({lang.native_name})")


def demo_script_analysis():
    """Demonstrate script analysis."""
    print("\n" + "=" * 60)
    print("6. Script Analysis")
    print("=" * 60)

    from apt.multilingual import Script, language_registry

    # List languages by script
    print("\nLanguages by script:")
    for script in [Script.LATIN, Script.CHINESE, Script.JAPANESE, Script.KOREAN]:
        langs = language_registry.list_languages(script=script.value)
        print(f"\n  {script.value.upper()}:")
        for lang in langs:
            print(f"    - {lang.name}")


def demo_config_integration():
    """Demonstrate integration with APTConfig."""
    print("\n" + "=" * 60)
    print("7. Configuration Integration")
    print("=" * 60)

    from apt.multilingual import get_language, get_vocab_size

    # Example: Creating configs for different languages
    print("\nRecommended configurations by language:")

    languages = ['en', 'zh', 'ja', 'ko']
    for code in languages:
        lang = get_language(code)
        vocab_size = get_vocab_size(code)

        print(f"\n  {lang.name}:")
        print(f"    Language code: {lang.code}")
        print(f"    Vocabulary size: {vocab_size:,}")
        print(f"    Script: {lang.script.value}")

        if lang.has_feature('needs_segmentation'):
            print(f"    Note: Requires word segmentation")

        if lang.has_feature('no_spaces'):
            print(f"    Note: No spaces between words")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("APT Multilingual Support Demo")
    print("=" * 60)

    try:
        demo_language_registry()
        demo_language_detection()
        demo_language_features()
        demo_rtl_languages()
        demo_language_groups()
        demo_script_analysis()
        demo_config_integration()

        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
