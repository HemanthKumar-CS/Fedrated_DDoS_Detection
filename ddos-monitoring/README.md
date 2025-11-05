# DDoS Detection Real-time Monitoring Dashboard

This is a modern, real-time DDoS detection monitoring dashboard built with **Vue.js + FastAPI**.

## Features

✨ **Real-time Updates** - WebSocket connections for instant data streaming
🎯 **Live Threat Detection** - Attack probability, packets/second, anomalies
📊 **Dynamic Charts** - Detection rate and packets visualization (Chart.js)
🚨 **Real-time Alerts** - Immediate notification of detected threats
💻 **System Metrics** - CPU, Memory, Connections monitoring
🔌 **WebSocket-based** - True real-time, no page reloads
🎨 **Modern Dark UI** - Professional dark theme with animations

## Project Structure

```
ddos-monitoring/
├── backend.py          # FastAPI server with WebSocket support
├── index.html          # Vue.js frontend (single-page app)
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── start.sh           # Startup script
```

## Installation & Setup

### 1. Install Backend Dependencies

```bash
cd ddos-monitoring
pip install -r requirements.txt
```

### 2. Run the Backend Server

```bash
python backend.py
```

The server will start on:
- 🌐 **API**: http://localhost:8000
- 🔌 **WebSocket**: ws://localhost:8000/ws/*
- 📚 **Docs**: http://localhost:8000/docs

### 3. Open the Dashboard

Simply open `index.html` in your browser:
```bash
# Windows
start index.html

# Mac
open index.html

# Linux
xdg-open index.html
```

Or access via HTTP server (recommended):
```bash
# Python 3
python -m http.server 5000

# Then visit http://localhost:5000
```

## API Endpoints

### REST API

```
GET  /health                  # Health check
GET  /api/detection-status    # Current detection status
GET  /api/alerts              # Recent alerts
GET  /api/metrics             # System metrics
```

### WebSocket Endpoints

```
WS   /ws/live-detection       # Real-time detection data (1 update/second)
WS   /ws/alerts               # Real-time alerts stream
WS   /ws/metrics              # Real-time system metrics (1 update/2 seconds)
```

## Real-time Updates

### Detection Data Updates
- **Detection Rate**: 0-100%
- **Threat Level**: LOW, MEDIUM, HIGH, CRITICAL
- **Packets/Second**: Real-time traffic count
- **Anomalies**: Number of detected anomalies
- **Is Under Attack**: Boolean attack status

### Alerts
- **ID**: Unique alert identifier
- **Message**: Alert description
- **Level**: critical, warning, info
- **Source**: Sensor or system identifier
- **Timestamp**: ISO format timestamp

### Metrics
- **CPU Usage**: 0-100%
- **Memory Usage**: 0-100%
- **Active Connections**: WebSocket connections
- **Packets Processed**: Total packets
- **Requests/Second**: Current throughput

## Dashboard Components

### Metrics Grid
- **Threat Level** - Current threat classification with visual indicator
- **Detection Rate** - Probability percentage with progress bar
- **Packets/Second** - Real-time network traffic
- **Anomalies** - Detected anomalies count
- **CPU Usage** - System CPU utilization
- **Memory Usage** - System memory utilization

### Charts
- **Detection Rate Chart** - 60-second history of detection rates
- **Packets/Second Chart** - 60-second history of packet traffic

### Alerts Section
- Real-time alert feed (latest first)
- Color-coded by severity (Critical, Warning, Info)
- Shows source and timestamp
- Auto-updates as alerts arrive

### Connection Status
- Real-time WebSocket connection indicator
- Shows connected/disconnected status
- Auto-reconnect on disconnect

## Attack Simulation

The dashboard simulates attacks based on time:
- **Every 15 minutes**: 5-minute attack window
- **During attack window**: 
  - High detection rates (70-95%)
  - High packet rates (1000-5000 packets/sec)
  - More anomalies detected
  - Critical alerts generated
- **Outside attack window**:
  - Low detection rates (5-20%)
  - Low packet rates (10-100 packets/sec)
  - Few anomalies
  - Info-level alerts only

## Features in Action

### Live Metrics
Metrics update in real-time via WebSocket connections:
- No page refresh needed
- Smooth animations
- Color-coded status indicators

### Dynamic Charts
Charts update automatically as new data arrives:
- 60-second sliding window
- Smooth line animations
- Bar charts for volume data

### Real-time Alerts
Alerts appear instantly at the top of the feed:
- Slide-in animation
- Color-coded severity levels
- Source and timestamp included

## Customization

### Change Update Frequencies
Edit `backend.py` WebSocket functions:
```python
await asyncio.sleep(1)  # Change this value for different intervals
```

### Modify Colors/Styling
Edit styles in `index.html` `<style>` section:
```css
.status-badge.critical {
    background: rgba(239, 68, 68, 0.1);
    border-color: #ef4444;
    color: #ef4444;
}
```

### Add New Metrics
1. Add new WebSocket endpoint in `backend.py`
2. Add metric ref in `index.html` metrics section
3. Update chart or display component

## Troubleshooting

### WebSocket Connection Fails
- Ensure backend is running on port 8000
- Check firewall settings
- Browser console shows connection errors

### Charts Not Rendering
- Verify Chart.js CDN is accessible
- Check browser console for errors
- Ensure canvas elements have proper IDs

### Styling Issues
- Clear browser cache
- Check CSS vendor prefixes for browser compatibility
- Ensure no CSS ad blockers interfering

## Performance Tips

- Keep browser console closed when not debugging
- Use modern browsers for best performance
- Disable unnecessary browser extensions
- Keep backend and frontend on same network for low latency

## Next Steps

To connect real attack data from your detection system:

1. **Modify `backend.py`** to read from your actual attack simulator
2. **Update WebSocket handlers** to stream real detection data
3. **Adjust thresholds** for threat level classification
4. **Add authentication** if needed for security

## Browser Support

- Chrome/Edge 88+
- Firefox 87+
- Safari 14+
- Opera 74+

## License

Part of the Federated DDoS Detection system.

---

**Happy Monitoring!** 🎯🛡️
