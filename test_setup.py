#!/usr/bin/env python3
"""
Test script to verify CrewAI setup with CodeInterpreterTool
"""

import sys
import warnings
warnings.filterwarnings("ignore")

def test_imports():
    """Test if all required packages can be imported."""
    try:
        from crewai import Agent, Task, Crew
        print("✅ CrewAI imported successfully")
        
        from crewai_tools import CodeInterpreterTool
        print("✅ CodeInterpreterTool imported successfully")
        
        from vnstock import Vnstock
        print("✅ vnstock imported successfully")
        
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_code_interpreter():
    """Test CodeInterpreterTool initialization."""
    try:
        from crewai_tools import CodeInterpreterTool
        
        # Initialize the tool
        interpreter = CodeInterpreterTool()
        print("✅ CodeInterpreterTool initialized successfully")
        
        # Test basic functionality
        agent = Agent(
            role="Test Analyst",
            goal="Test code execution",
            backstory="Test agent",
            tools=[interpreter],
            verbose=False
        )
        print("✅ Agent with CodeInterpreterTool created successfully")
        
        return True
    except Exception as e:
        print(f"❌ CodeInterpreterTool test failed: {e}")
        return False

def test_vnstock_connection():
    """Test basic vnstock functionality."""
    try:
        from vnstock import Vnstock
        
        # Test basic stock data fetch
        stock = Vnstock().stock(symbol="VIC", source="VCI")
        print("✅ vnstock connection successful")
        
        return True
    except Exception as e:
        print(f"❌ vnstock test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing CrewAI Data Analyst Setup...\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Code Interpreter Tests", test_code_interpreter),
        ("vnstock Connection Tests", test_vnstock_connection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   Failed - check installation")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to run the application.")
        print("\nTo start the application:")
        print("  streamlit run app.py")
    else:
        print("\n❌ Some tests failed. Please check the installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()
