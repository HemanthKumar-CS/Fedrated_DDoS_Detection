# Streamlit Monitoring App

This project is a Streamlit web application designed for live updates and visualization of system states, including detection status, alerts, and normal conditions.

## Project Structure

```
streamlit-monitoring-app
├── src
│   ├── app.py                  # Main entry point for the Streamlit application
│   ├── pages
│   │   ├── detection_status.py  # Displays current detection status
│   │   ├── alerts.py            # Interface for displaying alerts
│   │   └── system_state.py      # Shows overall system state
│   ├── components
│   │   ├── charts.py            # Functions for creating charts
│   │   ├── metrics.py           # Functions for displaying key metrics
│   │   └── status_indicators.py  # Functions for visual indicators
│   ├── utils
│   │   ├── data_processor.py     # Utility functions for data processing
│   │   └── config.py            # Configuration settings
│   └── data
│       └── sample_data.json     # Sample data for testing
├── requirements.txt              # Required Python packages
├── .streamlit
│   └── config.toml              # Streamlit configuration settings
└── README.md                     # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-monitoring-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command in your terminal:
```
streamlit run src/app.py
```

## Features

- **Detection Status**: View real-time detection data and trends.
- **Alerts**: Monitor alerts with a user-friendly interface.
- **System State Visualization**: Get an overview of the system's health, including normal conditions and alerts.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.