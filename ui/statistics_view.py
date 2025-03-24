import logging
import json
import datetime
from typing import Dict, Any, TYPE_CHECKING

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTabWidget, QGroupBox, QFormLayout,
                             QComboBox, QDateEdit, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QDate
from PyQt5.QtGui import QFont, QColor, QPalette, QPainter

# Only import AlertManager during type checking to avoid circular imports
if TYPE_CHECKING:
    from core.alert_manager import AlertManager

# For charts
try:
    from PyQt5.QtChart import (QChart, QChartView, QBarSet, QBarSeries,
                               QBarCategoryAxis, QValueAxis, QPieSeries)

    QTCHART_AVAILABLE = True
except ImportError:
    # If QtChart is not available, create stub classes
    QTCHART_AVAILABLE = False


    class QChart:
        pass


    class QChartView(QWidget):
        def __init__(self, chart=None, parent=None):
            super().__init__(parent)
            self.setMinimumSize(400, 300)


    class QBarSet:
        pass


    class QBarSeries:
        pass


    class QBarCategoryAxis:
        pass


    class QValueAxis:
        pass


    class QPieSeries:
        pass


    logging.warning("PyQt5.QtChart not available. Charts will not be displayed.")

logger = logging.getLogger("FOD.StatisticsView")


class StatisticsViewWidget(QWidget):
    """
    Widget to display statistical information and charts
    """

    def __init__(self, alert_manager: 'AlertManager', parent=None):
        super().__init__(parent)

        self.alert_manager = alert_manager
        self.stats = {}

        # Initialize UI
        self.init_ui()

        # Load initial data
        self.refresh()

    def init_ui(self):
        """Initialize UI components"""
        main_layout = QVBoxLayout(self)

        # Filter controls
        filter_layout = QHBoxLayout()

        # Date range
        date_group = QGroupBox("Date Range")
        date_layout = QHBoxLayout(date_group)

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addDays(-30))
        date_layout.addWidget(QLabel("From:"))
        date_layout.addWidget(self.start_date)

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        date_layout.addWidget(QLabel("To:"))
        date_layout.addWidget(self.end_date)

        filter_layout.addWidget(date_group)

        # Chart type
        chart_group = QGroupBox("Chart Type")
        chart_layout = QHBoxLayout(chart_group)

        self.chart_combo = QComboBox()
        self.chart_combo.addItem("Alerts by Day", "day")
        self.chart_combo.addItem("Alerts by Severity", "severity")
        self.chart_combo.addItem("Alerts by ROI", "roi")
        self.chart_combo.addItem("Top Classes", "classes")
        chart_layout.addWidget(self.chart_combo)

        filter_layout.addWidget(chart_group)

        # Filter button
        self.filter_button = QPushButton("Update")
        self.filter_button.clicked.connect(self.refresh)
        filter_layout.addWidget(self.filter_button)

        # Export button
        self.export_button = QPushButton("Export Report")
        self.export_button.clicked.connect(self.export_report)
        filter_layout.addWidget(self.export_button)

        main_layout.addLayout(filter_layout)

        # Create tab widget for different views
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Dashboard tab
        dash_tab = QWidget()
        dash_layout = QVBoxLayout(dash_tab)

        # Summary cards
        summary_layout = QHBoxLayout()

        # Total alerts card
        self.total_alerts_card = self._create_summary_card("Total Alerts", "0")
        summary_layout.addWidget(self.total_alerts_card)

        # Last 24h alerts card
        self.last_24h_card = self._create_summary_card("Last 24 Hours", "0")
        summary_layout.addWidget(self.last_24h_card)

        # Last 7 days alerts card
        self.last_7d_card = self._create_summary_card("Last 7 Days", "0")
        summary_layout.addWidget(self.last_7d_card)

        # High severity alerts card
        self.high_severity_card = self._create_summary_card("High Severity", "0")
        self.high_severity_card.setStyleSheet("background-color: rgba(255, 120, 120, 30%);")
        summary_layout.addWidget(self.high_severity_card)

        dash_layout.addLayout(summary_layout)

        # Main chart
        self.chart_container = QWidget()
        chart_container_layout = QVBoxLayout(self.chart_container)

        self.chart_title = QLabel("Alerts by Day")
        self.chart_title.setAlignment(Qt.AlignCenter)
        font = self.chart_title.font()
        font.setPointSize(14)
        font.setBold(True)
        self.chart_title.setFont(font)
        chart_container_layout.addWidget(self.chart_title)

        # Create chart view
        self.chart_view = self._create_chart_view()
        chart_container_layout.addWidget(self.chart_view)

        dash_layout.addWidget(self.chart_container)

        # Charts tab
        charts_tab = QWidget()
        charts_layout = QVBoxLayout(charts_tab)

        # Create multiple chart views
        charts_row1 = QHBoxLayout()

        # Severity distribution chart
        severity_group = QGroupBox("Alerts by Severity")
        severity_layout = QVBoxLayout(severity_group)
        self.severity_chart_view = self._create_chart_view()
        severity_layout.addWidget(self.severity_chart_view)
        charts_row1.addWidget(severity_group)

        # ROI distribution chart
        roi_group = QGroupBox("Alerts by ROI")
        roi_layout = QVBoxLayout(roi_group)
        self.roi_chart_view = self._create_chart_view()
        roi_layout.addWidget(self.roi_chart_view)
        charts_row1.addWidget(roi_group)

        charts_layout.addLayout(charts_row1)

        charts_row2 = QHBoxLayout()

        # Top classes chart
        classes_group = QGroupBox("Top Detected Classes")
        classes_layout = QVBoxLayout(classes_group)
        self.classes_chart_view = self._create_chart_view()
        classes_layout.addWidget(self.classes_chart_view)
        charts_row2.addWidget(classes_group)

        # Daily trend chart
        trend_group = QGroupBox("Alert Trend")
        trend_layout = QVBoxLayout(trend_group)
        self.trend_chart_view = self._create_chart_view()
        trend_layout.addWidget(self.trend_chart_view)
        charts_row2.addWidget(trend_group)

        charts_layout.addLayout(charts_row2)

        # Add tabs
        self.tabs.addTab(dash_tab, "Dashboard")
        self.tabs.addTab(charts_tab, "Detailed Charts")

    def _create_summary_card(self, title: str, value: str) -> QGroupBox:
        """Create a summary card widget"""
        card = QGroupBox(title)
        card.setStyleSheet("""
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
                background-color: rgba(240, 240, 240, 50%);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 5px;
            }
        """)

        layout = QVBoxLayout(card)

        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignCenter)
        font = value_label.font()
        font.setPointSize(24)
        font.setBold(True)
        value_label.setFont(font)

        layout.addWidget(value_label)

        return card

    def _create_chart_view(self) -> QChartView:
        """Create a chart view widget"""
        if not QTCHART_AVAILABLE:
            placeholder = QLabel("QtChart not available. Charts cannot be displayed.")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
            placeholder.setMinimumSize(400, 300)
            return placeholder

        # Create empty chart
        chart = QChart()
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)

        # Create chart view
        chart_view = QChartView(chart)
        # Use QPainter.Antialiasing instead of chart_view.Antialiasing
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setMinimumSize(400, 300)

        return chart_view

    def refresh(self):
        """Refresh statistics data and update charts"""
        # Get statistics from alert manager
        self.stats = self.alert_manager.get_statistics()

        # Update summary cards
        self.update_summary_cards()

        # Update charts based on selected type
        if QTCHART_AVAILABLE:
            chart_type = self.chart_combo.currentData()

            if chart_type == "day":
                self.chart_title.setText("Alerts by Day")
                self._update_day_chart(self.chart_view)
            elif chart_type == "severity":
                self.chart_title.setText("Alerts by Severity")
                self._update_severity_chart(self.chart_view)
            elif chart_type == "roi":
                self.chart_title.setText("Alerts by ROI")
                self._update_roi_chart(self.chart_view)
            elif chart_type == "classes":
                self.chart_title.setText("Top Detected Classes")
                self._update_classes_chart(self.chart_view)

            # Update detail charts
            self._update_severity_chart(self.severity_chart_view)
            self._update_roi_chart(self.roi_chart_view)
            self._update_classes_chart(self.classes_chart_view)
            self._update_day_chart(self.trend_chart_view)

    def update_summary_cards(self):
        """Update the summary cards with current statistics"""
        # Total alerts
        total_alerts = self.stats.get("total_alerts", 0)
        self._update_card_value(self.total_alerts_card, str(total_alerts))

        # Last 24 hours
        last_24h = self.stats.get("count_last_hour", 0)  # This is actually count_last_day from alert_manager
        self._update_card_value(self.last_24h_card, str(last_24h))

        # Last 7 days
        # Sum the last 7 days from alerts_by_day
        last_7d = 0
        days_data = self.stats.get("alerts_by_day", {})
        today = datetime.date.today()

        for i in range(7):
            day = (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            last_7d += days_data.get(day, 0)

        self._update_card_value(self.last_7d_card, str(last_7d))

        # High severity
        high_severity = self.stats.get("alerts_by_severity", {}).get(3, 0)
        self._update_card_value(self.high_severity_card, str(high_severity))

    def _update_card_value(self, card: QGroupBox, value: str):
        """Update the value displayed in a card"""
        # Find the value label (first QLabel in the layout)
        for i in range(card.layout().count()):
            widget = card.layout().itemAt(i).widget()
            if isinstance(widget, QLabel):
                widget.setText(value)
                break

    def _update_day_chart(self, chart_view):
        """Update the day chart with current data"""
        if not QTCHART_AVAILABLE or not isinstance(chart_view, QChartView):
            return

        # Get data
        days_data = self.stats.get("alerts_by_day", {})

        # Create a new chart
        chart = QChart()
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.legend().setVisible(False)

        if not days_data:
            chart.setTitle("No data available")
            chart_view.setChart(chart)
            return

        # Sort days
        sorted_days = sorted(days_data.keys())

        # Create bar set
        bar_set = QBarSet("Alerts")

        # Add data to bar set
        for day in sorted_days:
            bar_set.append(days_data[day])

        # Create bar series
        series = QBarSeries()
        series.append(bar_set)

        # Add series to chart
        chart.addSeries(series)

        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(sorted_days)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        max_value = max(days_data.values()) if days_data else 0
        axis_y.setRange(0, max_value * 1.1)  # Add some margin
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        # Set chart title
        chart.setTitle(f"Alerts by Day ({len(sorted_days)} days)")

        # Update chart view
        chart_view.setChart(chart)

    def _update_severity_chart(self, chart_view):
        """Update the severity chart with current data"""
        if not QTCHART_AVAILABLE or not isinstance(chart_view, QChartView):
            return

        # Get data
        severity_data = self.stats.get("alerts_by_severity", {})

        # Create a new chart
        chart = QChart()
        chart.setAnimationOptions(QChart.SeriesAnimations)

        if not severity_data:
            chart.setTitle("No data available")
            chart_view.setChart(chart)
            return

        # Create pie series
        series = QPieSeries()

        # Define severity labels
        severity_labels = {
            1: "Low",
            2: "Medium",
            3: "High"
        }

        # Add slices
        for severity, count in severity_data.items():
            if count > 0:
                severity_key = int(severity)
                label = severity_labels.get(severity_key, f"Unknown ({severity})")
                slice = series.append(f"{label} ({count})", count)

                # Set slice colors
                if severity_key == 1:
                    slice.setBrush(QColor(100, 149, 237))  # Blue
                elif severity_key == 2:
                    slice.setBrush(QColor(255, 165, 0))  # Orange
                elif severity_key == 3:
                    slice.setBrush(QColor(220, 20, 60))  # Red
                    slice.setExploded(True)

        # Add series to chart
        chart.addSeries(series)

        # Set chart title
        total = sum(severity_data.values())
        chart.setTitle(f"Alerts by Severity (Total: {total})")

        # Update chart view
        chart_view.setChart(chart)

    def _update_roi_chart(self, chart_view):
        """Update the ROI chart with current data"""
        if not QTCHART_AVAILABLE or not isinstance(chart_view, QChartView):
            return

        # Get data
        roi_data = self.stats.get("alerts_by_roi", {})

        # Create a new chart
        chart = QChart()
        chart.setAnimationOptions(QChart.SeriesAnimations)

        if not roi_data:
            chart.setTitle("No data available")
            chart_view.setChart(chart)
            return

        # Create bar set
        bar_set = QBarSet("Alerts")

        # Limit to top 10 ROIs
        sorted_rois = sorted(roi_data.items(), key=lambda x: x[1], reverse=True)[:10]
        roi_names = []

        # Add data to bar set
        for roi_name, count in sorted_rois:
            bar_set.append(count)
            roi_names.append(roi_name)

        # Create bar series
        series = QBarSeries()
        series.append(bar_set)

        # Add series to chart
        chart.addSeries(series)

        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(roi_names)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        max_value = max(roi_data.values()) if roi_data else 0
        axis_y.setRange(0, max_value * 1.1)  # Add some margin
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        # Set chart title
        chart.setTitle(f"Alerts by ROI (Top {len(sorted_rois)})")

        # Update chart view
        chart_view.setChart(chart)

    def _update_classes_chart(self, chart_view):
        """Update the classes chart with current data"""
        if not QTCHART_AVAILABLE or not isinstance(chart_view, QChartView):
            return

        # Get data
        classes_data = self.stats.get("top_classes", {})

        # Create a new chart
        chart = QChart()
        chart.setAnimationOptions(QChart.SeriesAnimations)

        if not classes_data:
            chart.setTitle("No data available")
            chart_view.setChart(chart)
            return

        # Create bar set
        bar_set = QBarSet("Count")

        # Limit to top 10 classes
        sorted_classes = sorted(classes_data.items(), key=lambda x: x[1], reverse=True)[:10]
        class_names = []

        # Add data to bar set
        for class_name, count in sorted_classes:
            bar_set.append(count)
            class_names.append(class_name)

        # Create bar series
        series = QBarSeries()
        series.append(bar_set)

        # Add series to chart
        chart.addSeries(series)

        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(class_names)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        max_value = max(classes_data.values()) if classes_data else 0
        axis_y.setRange(0, max_value * 1.1)  # Add some margin
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        # Set chart title
        chart.setTitle(f"Top Detected Classes (Top {len(sorted_classes)})")

        # Update chart view
        chart_view.setChart(chart)

    def export_report(self):
        """Export statistics as a PDF or HTML report"""
        try:
            from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
            from PyQt5.QtGui import QPainter
            from PyQt5.QtCore import QSize, QRect
            from PyQt5.QtWidgets import QFileDialog
        except ImportError:
            QMessageBox.warning(self, "Export Error", "Qt print support not available. Cannot export report.")
            return

        # Ask for file location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report",
            "fod_detection_report.pdf",
            "PDF Files (*.pdf)"
        )

        if not file_path:
            return

        try:
            # Create printer
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(file_path)
            printer.setPageSize(QPrinter.A4)

            # Start painter
            painter = QPainter()
            painter.begin(printer)

            # Calculate page metrics
            page_rect = printer.pageRect()
            width = page_rect.width()
            height = page_rect.height()

            # Draw title
            font = painter.font()
            font.setPointSize(18)
            font.setBold(True)
            painter.setFont(font)

            title_rect = QRect(0, 0, width, 50)
            painter.drawText(title_rect, Qt.AlignCenter, "FOD Detection System - Statistics Report")

            # Draw date range
            font.setPointSize(10)
            font.setBold(False)
            painter.setFont(font)

            date_text = f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            date_rect = QRect(0, 50, width, 30)
            painter.drawText(date_rect, Qt.AlignCenter, date_text)

            # Draw summary statistics
            font.setPointSize(14)
            font.setBold(True)
            painter.setFont(font)

            painter.drawText(QRect(50, 100, width - 100, 30), Qt.AlignLeft, "Summary Statistics")

            # Draw summary metrics
            font.setPointSize(10)
            font.setBold(False)
            painter.setFont(font)

            y_pos = 140
            metrics = [
                f"Total Alerts: {self.stats.get('total_alerts', 0)}",
                f"Alerts in Last 24 Hours: {self.stats.get('count_last_hour', 0)}",
                f"High Severity Alerts: {self.stats.get('alerts_by_severity', {}).get(3, 0)}",
                f"Medium Severity Alerts: {self.stats.get('alerts_by_severity', {}).get(2, 0)}",
                f"Low Severity Alerts: {self.stats.get('alerts_by_severity', {}).get(1, 0)}"
            ]

            for metric in metrics:
                painter.drawText(QRect(70, y_pos, width - 140, 20), Qt.AlignLeft, metric)
                y_pos += 25

            # Draw charts
            font.setPointSize(14)
            font.setBold(True)
            painter.setFont(font)

            painter.drawText(QRect(50, 280, width - 100, 30), Qt.AlignLeft, "Alert Distribution")

            # Render severity chart
            if QTCHART_AVAILABLE and hasattr(self, 'severity_chart_view') and isinstance(self.severity_chart_view,
                                                                                         QChartView):
                chart_size = QSize(width / 2 - 70, 300)
                self.severity_chart_view.chart().setBackgroundVisible(False)
                self.severity_chart_view.render(painter, QRect(50, 320, chart_size.width(), chart_size.height()))

            # Render ROI chart
            if QTCHART_AVAILABLE and hasattr(self, 'roi_chart_view') and isinstance(self.roi_chart_view, QChartView):
                chart_size = QSize(width / 2 - 70, 300)
                self.roi_chart_view.chart().setBackgroundVisible(False)
                self.roi_chart_view.render(painter, QRect(width / 2 + 20, 320, chart_size.width(), chart_size.height()))

            # Move to next section
            font.setPointSize(14)
            font.setBold(True)
            painter.setFont(font)

            painter.drawText(QRect(50, 630, width - 100, 30), Qt.AlignLeft, "Trend Analysis")

            # Render trend chart
            if QTCHART_AVAILABLE and hasattr(self, 'trend_chart_view') and isinstance(self.trend_chart_view,
                                                                                      QChartView):
                chart_size = QSize(width - 100, 300)
                self.trend_chart_view.chart().setBackgroundVisible(False)
                self.trend_chart_view.render(painter, QRect(50, 670, chart_size.width(), chart_size.height()))

            # Finish painting
            painter.end()

            QMessageBox.information(self, "Export Successful", f"Report exported to {file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Error exporting report: {e}")