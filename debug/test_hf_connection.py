#!/usr/bin/env python3
"""
Test script to verify Hugging Face connection and repository creation
"""

from huggingface_hub import HfApi, create_repo
import subprocess

def check_hf_login():
    """Check if user is logged into Hugging Face"""
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], 
                               capture_output=True, text=True)
        if "Not logged in" in result.stdout:
            print("❌ You need to log into Hugging Face first!")
            print("Run: huggingface-cli login")
            return False
        else:
            print(f"✅ Logged in to Hugging Face: {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"❌ Error checking HF login: {e}")
        return False

def test_repository_creation():
    """Test repository creation with proper naming"""
    
    if not check_hf_login():
        return False
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Get current user info
        user_info = api.whoami()
        username = user_info['name']
        print(f"👤 Current user: {username}")
        
        # Test repository name
        test_repo_name = "test-hf-connection"
        full_repo_id = f"{username}/{test_repo_name}"
        
        print(f"🧪 Testing repository creation: {full_repo_id}")
        
        # Try to create a test repository
        create_repo(
            repo_id=full_repo_id,
            token=api.token,
            repo_type="dataset",
            private=True,  # Make it private for testing
            exist_ok=True
        )
        
        print(f"✅ Repository creation successful!")
        print(f"🔗 Repository URL: https://huggingface.co/datasets/{full_repo_id}")
        
        # Test a simple file upload
        print("🧪 Testing file upload...")
        
        # Create a simple test file
        with open("test_file.txt", "w") as f:
            f.write("This is a test file for HF connection verification.\n")
        
        api.upload_file(
            path_or_fileobj="test_file.txt",
            path_in_repo="test_file.txt",
            repo_id=full_repo_id,
            repo_type="dataset",
            commit_message="Test file upload"
        )
        
        print("✅ File upload successful!")
        
        # Clean up test file
        import os
        os.remove("test_file.txt")
        
        print("\n🎉 All tests passed! Your HF setup is working correctly.")
        print("✅ You can now run the main backup script safely.")
        
        # Ask if user wants to delete the test repository
        delete_choice = input("\nDelete test repository? (y/N): ").strip().lower()
        if delete_choice in ['y', 'yes']:
            try:
                api.delete_repo(repo_id=full_repo_id, repo_type="dataset")
                print("✅ Test repository deleted")
            except Exception as e:
                print(f"❌ Could not delete test repository: {e}")
                print(f"📝 You can manually delete it at: https://huggingface.co/datasets/{full_repo_id}/settings")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def main():
    print("🔧 Hugging Face Connection Test")
    print("=" * 40)
    print("This script will:")
    print("  1. Verify HF authentication")
    print("  2. Test repository creation")
    print("  3. Test file upload")
    print("  4. Clean up test files")
    print()
    
    success = test_repository_creation()
    
    if success:
        print("\n✅ Connection test completed successfully!")
    else:
        print("\n❌ Connection test failed!")
        print("🔧 Please check your Hugging Face authentication and try again.")

if __name__ == "__main__":
    main() 