"""
Template System Verification Script

Tests the template system end-to-end:
1. Loading templates from JSON
2. Matching templates to briefs
3. Generating designs from templates
"""

import os
from PIL import Image
from src.designers.enhanced_compositional_designer import EnhancedCompositionalDesigner

def test_template_loading():
    """Test 1: Verify templates load correctly."""
    print("\n" + "="*60)
    print("TEST 1: Template Loading")
    print("="*60)

    designer = EnhancedCompositionalDesigner()

    templates = designer.template_library.templates
    print(f"✅ Loaded {len(templates)} templates")

    for tid, template in templates.items():
        print(f"   - {template.name} ({template.category})")
        print(f"     Tags: {', '.join(template.tags)}")
        print(f"     Elements: {len(template.layout.elements)}, Typography: {len(template.layout.typography)}")
        print(f"     Fixed: {template.is_fixed}")
        print()

    return len(templates) > 0

def test_template_matching():
    """Test 2: Verify template matching logic."""
    print("\n" + "="*60)
    print("TEST 2: Template Matching")
    print("="*60)

    designer = EnhancedCompositionalDesigner()

    test_briefs = [
        {
            "description": "instagram post about mindfulness quote",
            "format": "instagram",
            "keywords": ["quote", "minimal"]
        },
        {
            "description": "corporate event poster for tech conference",
            "format": "poster",
            "keywords": ["event", "business"]
        },
        {
            "description": "company announcement banner for linkedin",
            "format": "linkedin",
            "keywords": ["announcement", "news"]
        }
    ]

    all_matched = True
    for brief in test_briefs:
        print(f"\nBrief: {brief['description']}")
        template, score = designer.template_matcher.match_template(brief)
        if template:
            print(f"✅ Matched: {template.name} (Score: {score:.2f})")
        else:
            print("❌ No match found")
            all_matched = False

    return all_matched

def test_template_generation():
    """Test 3: Generate actual designs from templates."""
    print("\n" + "="*60)
    print("TEST 3: Template-Based Design Generation")
    print("="*60)

    designer = EnhancedCompositionalDesigner()

    test_cases = [
        {
            "brief": "Create an inspiring quote post for Instagram about innovation",
            "template_id": "insta_minimal_quote",
            "output": "data/template_test_instagram.png"
        },
        {
            "brief": "Design a corporate event poster for AI Summit 2025",
            "template_id": "corp_event_poster",
            "output": "data/template_test_corporate.png"
        },
        {
            "brief": "Create an elegant balanced quote from a CEO",
            "template_id": "real_balanced_center",
            "output": "data/template_real_balanced.png"
        },
        {
            "brief": "Modern story overlay for product launch",
            "template_id": "real_bottom_anchor",
            "output": "data/template_real_bottom.png"
        }
    ]

    all_generated = True
    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test['brief']}")
        print(f"    Using template: {test['template_id']}")

        try:
            design = designer.design_from_template(
                test["brief"],
                template_id=test["template_id"]
            )

            # Save output
            design.save(test["output"])
            print(f"    ✅ Saved to: {test['output']}")

        except Exception as e:
            print(f"    ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            all_generated = False

    return all_generated

def test_auto_matching():
    """Test 4: Auto-match without explicit template ID."""
    print("\n" + "="*60)
    print("TEST 4: Auto-Matching (No Template ID)")
    print("="*60)

    designer = EnhancedCompositionalDesigner()

    brief = "Create a professional announcement for our new product launch on LinkedIn"
    print(f"\nBrief: {brief}")

    try:
        design = designer.design_from_template(brief)
        output = "data/template_test_auto_match.png"
        design.save(output)
        print(f"✅ Auto-matched and generated: {output}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all verification tests."""
    print("\n" + "🧪" * 30)
    print("TEMPLATE SYSTEM VERIFICATION")
    print("🧪" * 30)

    results = []

    results.append(("Template Loading", test_template_loading()))
    results.append(("Template Matching", test_template_matching()))
    results.append(("Template Generation", test_template_generation()))
    results.append(("Auto-Matching", test_auto_matching()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests passed! Template system is ready.")
    else:
        print("\n⚠️ Some tests failed. Please review errors above.")

    return all_passed

if __name__ == "__main__":
    run_all_tests()
