"""
Data Download Script for Telco Customer Churn Dataset
Automatically downloads dataset from Kaggle or provides manual instructions
"""

import os
import sys
from pathlib import Path
import zipfile
import shutil

def setup_kaggle_api():
    """
    Setup Kaggle API credentials
    Returns True if successful, False otherwise
    """
    kaggle_json_path = Path.home() / '.kaggle' / 'kaggle.json'
    
    if kaggle_json_path.exists():
        print("✓ Kaggle API credentials found")
        return True
    else:
        print("\n⚠️  Kaggle API credentials not found")
        print("\nTo setup Kaggle API:")
        print("1. Go to https://www.kaggle.com/charankarthik14")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Move downloaded kaggle.json to ~/.kaggle/")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

def download_with_kaggle_api():
    """Download dataset using Kaggle API"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("\n📥 Downloading dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()
        
        # Create data directory
        data_dir = Path(__file__).parent / 'raw'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        dataset_name = 'blastchar/telco-customer-churn'
        api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
        
        print("✓ Dataset downloaded successfully!")
        print(f"📁 Location: {data_dir}")
        
        # List downloaded files
        files = list(data_dir.glob('*.csv'))
        if files:
            print(f"\n📄 Files downloaded:")
            for file in files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name} ({size_mb:.2f} MB)")
        
        return True
        
    except ImportError:
        print("\n❌ Kaggle API not installed")
        print("Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {str(e)}")
        return False

def manual_download_instructions():
    """Provide manual download instructions"""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\n1. Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    print("\n2. Click 'Download' button (requires Kaggle account)")
    print("\n3. Extract the downloaded ZIP file")
    print("\n4. Copy 'WA_Fn-UseC_-Telco-Customer-Churn.csv' to:")
    
    data_dir = Path(__file__).parent / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"   {data_dir.absolute()}")
    
    print("\n5. Verify file exists:")
    print(f"   python -c \"import pandas as pd; df = pd.read_csv('{data_dir}/WA_Fn-UseC_-Telco-Customer-Churn.csv'); print(f'✓ Dataset loaded: {{len(df)}} rows')\"")
    print("\n" + "="*60)

def verify_dataset():
    """Verify dataset was downloaded correctly"""
    data_dir = Path(__file__).parent / 'raw'
    expected_file = data_dir / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    if expected_file.exists():
        print("\n✓ Dataset verification successful!")
        
        # Load and verify
        try:
            import pandas as pd
            df = pd.read_csv(expected_file)
            
            print(f"\n📊 Dataset Summary:")
            print(f"   - Rows: {len(df):,}")
            print(f"   - Columns: {len(df.columns)}")
            print(f"   - Size: {expected_file.stat().st_size / (1024*1024):.2f} MB")
            print(f"\n   Columns: {', '.join(df.columns[:5])}...")
            
            return True
        except Exception as e:
            print(f"\n⚠️  Dataset exists but couldn't be loaded: {str(e)}")
            return False
    else:
        print(f"\n❌ Dataset not found at: {expected_file}")
        return False

def main():
    """Main execution flow"""
    print("="*60)
    print("TELCO CUSTOMER CHURN - DATASET DOWNLOAD")
    print("="*60)
    
    # Check if dataset already exists
    if verify_dataset():
        print("\n🎉 Dataset already downloaded and verified!")
        print("No action needed.")
        return
    
    # Try Kaggle API download
    if setup_kaggle_api():
        if download_with_kaggle_api():
            verify_dataset()
            return
    
    # Fallback to manual instructions
    manual_download_instructions()
    
    print("\n💡 Tip: After manual download, run this script again to verify")

if __name__ == "__main__":
    main()
