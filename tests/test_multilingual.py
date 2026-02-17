#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for APT Multilingual Support

Validates:
1. Language definitions and registry
2. Language detection
3. Script detection
4. Language features
5. TokenizerProvider interface
"""

import sys
import traceback


def test_imports():
    """Test that all multilingual modules can be imported."""
    print("=" * 60)
    print("Test 1: Multilingual Module Imports")
    print("=" * 60)

    try:
        from apt.multilingual import (
            Language,
            Script,
            Direction,
            LanguageFeatures,
            get_language,
            list_languages,
            detect_language,
            TokenizerProvider,
            language_registry,
            language_detector
        )
        print("✅ All multilingual modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_language_registry():
    """Test language registry functionality."""
    print("\n" + "=" * 60)
    print("Test 2: Language Registry")
    print("=" * 60)

    try:
        from apt.multilingual import get_language, list_languages, language_registry

        # Test getting language by code
        chinese = get_language('zh')
        assert chinese is not None
        assert chinese.code == 'zh'
        assert chinese.name == 'Chinese (Simplified)'
        print(f"✅ Get language by code: {chinese.name}")

        # Test getting language by name
        japanese = get_language('Japanese')
        assert japanese is not None
        assert japanese.code == 'ja'
        print(f"✅ Get language by name: {japanese.name}")

        # Test getting language by alias
        chinese2 = get_language('zh-cn')
        assert chinese2 is not None
        assert chinese2.code == 'zh'
        print(f"✅ Get language by alias: zh-cn → {chinese2.code}")

        # Test listing languages
        languages = list_languages()
        assert len(languages) > 0
        print(f"✅ List languages: {len(languages)} languages registered")

        # Test language count
        count = len(language_registry)
        assert count >= 12  # At least 12 predefined languages
        print(f"✅ Registry contains {count} languages")

        return True

    except Exception as e:
        print(f"❌ Language registry test failed: {e}")
        traceback.print_exc()
        return False


def test_language_detection():
    """Test language detection functionality."""
    print("\n" + "=" * 60)
    print("Test 3: Language Detection")
    print("=" * 60)

    try:
        from apt.multilingual import detect_language, detect_script, Script

        # Test English detection
        lang = detect_language("Hello, world!")
        assert lang == 'en'
        print(f"✅ English detected: {lang}")

        # Test Chinese detection
        lang = detect_language("你好世界")
        assert lang == 'zh'
        print(f"✅ Chinese detected: {lang}")

        # Test Japanese detection
        lang = detect_language("こんにちは")
        assert lang == 'ja'
        print(f"✅ Japanese detected: {lang}")

        # Test Korean detection
        lang = detect_language("안녕하세요")
        assert lang == 'ko'
        print(f"✅ Korean detected: {lang}")

        # Test script detection
        script = detect_script("Hello")
        assert script == Script.LATIN
        print(f"✅ Latin script detected")

        script = detect_script("你好")
        assert script == Script.CHINESE
        print(f"✅ Chinese script detected")

        return True

    except Exception as e:
        print(f"❌ Language detection test failed: {e}")
        traceback.print_exc()
        return False


def test_language_features():
    """Test language features."""
    print("\n" + "=" * 60)
    print("Test 4: Language Features")
    print("=" * 60)

    try:
        from apt.multilingual import get_language, LanguageFeatures

        # Test English features
        english = get_language('en')
        assert english.has_feature(LanguageFeatures.SPACES)
        assert english.has_feature(LanguageFeatures.ALPHABETIC)
        assert not english.has_feature(LanguageFeatures.TONES)
        print(f"✅ English features validated")

        # Test Chinese features
        chinese = get_language('zh')
        assert chinese.has_feature(LanguageFeatures.TONES)
        assert chinese.has_feature(LanguageFeatures.LOGOGRAPHIC)
        assert chinese.has_feature(LanguageFeatures.NEEDS_SEGMENTATION)
        assert not chinese.has_feature(LanguageFeatures.SPACES)
        print(f"✅ Chinese features validated")

        # Test Japanese features
        japanese = get_language('ja')
        assert japanese.has_feature(LanguageFeatures.MIXED_SCRIPT)
        assert japanese.has_feature(LanguageFeatures.SYLLABIC)
        assert japanese.has_feature(LanguageFeatures.SOV)
        print(f"✅ Japanese features validated")

        return True

    except Exception as e:
        print(f"❌ Language features test failed: {e}")
        traceback.print_exc()
        return False


def test_script_analysis():
    """Test script analysis."""
    print("\n" + "=" * 60)
    print("Test 5: Script Analysis")
    print("=" * 60)

    try:
        from apt.multilingual import language_detector, Script

        # Test mixed language detection
        mixed_text = "Hello 你好 world 世界"
        is_mixed = language_detector.is_mixed_language(mixed_text)
        assert is_mixed == True
        print(f"✅ Mixed language detected: {mixed_text}")

        # Test script distribution
        dist = language_detector.get_script_distribution(mixed_text)
        # Distribution returns script names as strings, not Script objects
        assert 'latin' in dist or Script.LATIN.value in dist
        assert 'chinese' in dist or Script.CHINESE.value in dist
        print(f"✅ Script distribution calculated: {len(dist)} scripts")

        return True

    except Exception as e:
        print(f"❌ Script analysis test failed: {e}")
        traceback.print_exc()
        return False


def test_vocab_size():
    """Test vocabulary size recommendations."""
    print("\n" + "=" * 60)
    print("Test 6: Vocabulary Size Recommendations")
    print("=" * 60)

    try:
        from apt.multilingual import get_vocab_size

        # Test vocab sizes
        en_vocab = get_vocab_size('en')
        zh_vocab = get_vocab_size('zh')
        ja_vocab = get_vocab_size('ja')

        assert en_vocab == 50000
        assert zh_vocab == 21128
        assert ja_vocab == 32000

        print(f"✅ English vocab size: {en_vocab:,}")
        print(f"✅ Chinese vocab size: {zh_vocab:,}")
        print(f"✅ Japanese vocab size: {ja_vocab:,}")

        return True

    except Exception as e:
        print(f"❌ Vocab size test failed: {e}")
        traceback.print_exc()
        return False


def test_rtl_languages():
    """Test RTL language support."""
    print("\n" + "=" * 60)
    print("Test 7: RTL Language Support")
    print("=" * 60)

    try:
        from apt.multilingual import language_registry, Direction

        # Test RTL languages
        rtl_langs = language_registry.get_rtl_languages()
        assert len(rtl_langs) > 0
        print(f"✅ Found {len(rtl_langs)} RTL language(s)")

        # Test Arabic
        arabic = language_registry.get_language('ar')
        assert arabic is not None
        assert arabic.direction == Direction.RTL
        print(f"✅ Arabic is RTL: {arabic.native_name}")

        return True

    except Exception as e:
        print(f"❌ RTL language test failed: {e}")
        traceback.print_exc()
        return False


def test_language_serialization():
    """Test language serialization."""
    print("\n" + "=" * 60)
    print("Test 8: Language Serialization")
    print("=" * 60)

    try:
        from apt.multilingual import get_language, Language

        # Test to_dict
        chinese = get_language('zh')
        lang_dict = chinese.to_dict()
        assert 'code' in lang_dict
        assert 'name' in lang_dict
        assert 'vocab_size' in lang_dict
        print(f"✅ Language serialized to dict: {len(lang_dict)} fields")

        # Test from_dict
        chinese2 = Language.from_dict(lang_dict)
        assert chinese2.code == chinese.code
        assert chinese2.name == chinese.name
        print(f"✅ Language deserialized from dict")

        return True

    except Exception as e:
        print(f"❌ Language serialization test failed: {e}")
        traceback.print_exc()
        return False


def test_language_groups():
    """Test predefined language groups."""
    print("\n" + "=" * 60)
    print("Test 9: Predefined Language Groups")
    print("=" * 60)

    try:
        from apt.multilingual import (
            EAST_ASIAN_LANGUAGES,
            EUROPEAN_LANGUAGES,
            RTL_LANGUAGES
        )

        # Test East Asian languages
        assert len(EAST_ASIAN_LANGUAGES) == 4
        codes = [lang.code for lang in EAST_ASIAN_LANGUAGES]
        assert 'zh' in codes
        assert 'ja' in codes
        assert 'ko' in codes
        print(f"✅ East Asian languages: {len(EAST_ASIAN_LANGUAGES)}")

        # Test European languages
        assert len(EUROPEAN_LANGUAGES) >= 5
        codes = [lang.code for lang in EUROPEAN_LANGUAGES]
        assert 'en' in codes
        assert 'fr' in codes
        assert 'de' in codes
        print(f"✅ European languages: {len(EUROPEAN_LANGUAGES)}")

        # Test RTL languages
        assert len(RTL_LANGUAGES) >= 1
        codes = [lang.code for lang in RTL_LANGUAGES]
        assert 'ar' in codes
        print(f"✅ RTL languages: {len(RTL_LANGUAGES)}")

        return True

    except Exception as e:
        print(f"❌ Language groups test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("APT Multilingual Support Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Language Registry", test_language_registry),
        ("Language Detection", test_language_detection),
        ("Language Features", test_language_features),
        ("Script Analysis", test_script_analysis),
        ("Vocabulary Size", test_vocab_size),
        ("RTL Languages", test_rtl_languages),
        ("Serialization", test_language_serialization),
        ("Language Groups", test_language_groups),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    print("\n" + "=" * 60)
    if passed_count == total_count:
        print(f"✅ All {total_count} tests PASSED")
        print("=" * 60)
        return 0
    else:
        print(f"❌ {total_count - passed_count}/{total_count} tests FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
