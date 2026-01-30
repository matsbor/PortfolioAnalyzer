import pandas as pd
import yfinance as yf

def generate_sector_map():
    # Adding broad market miners to ensure diversification
    data = {
        "Symbol": [
            "AEM", "PAAS", "VZLA", "MAG", "SILV", "LMNR", "DSV.V", "SKE.TO", 
            "GFI", "AU", "KGC", "EQX", "NGD", "HBM", "CDE", "HL", "AG", "FSM",
            "SSR.TO", "K.TO", "IMG.TO", "BTO.TO", "WPM", "FNV", "RGLD", "MTA"
        ],
        "Metal_Type": ["Gold", "Silver", "Silver", "Silver", "Silver", "Gold", "Silver", "Gold", "Gold", "Gold", "Gold", "Gold", "Gold", "Gold", "Silver", "Silver", "Silver", "Silver", "Gold", "Gold", "Gold", "Gold", "Silver", "Gold", "Gold", "Gold"],
        "Jurisdiction": ["Canada", "Mexico", "Mexico", "Mexico", "Mexico", "USA", "Mexico", "Canada", "Africa", "Africa", "USA", "USA", "Canada", "Canada", "USA", "USA", "Mexico", "Mexico", "USA", "Canada", "Canada", "Africa", "Global", "Global", "Global", "Global"]
    }
    df = pd.DataFrame(data)
    df.to_csv("master_discovery_list.csv", index=False)
    print("🚀 Sovereign Hunter list expanded with 26+ diversifed symbols.")

if __name__ == "__main__":
    generate_sector_map()
