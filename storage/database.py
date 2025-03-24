import sqlite3
import logging
import os
import json
import csv
import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import class manager
from storage.class_manager import ClassManager

logger = logging.getLogger("FOD.Database")


class AlertDatabase:
    """
    SQLite database manager for storing and retrieving alert information
    """

    # Class manager instance
    _class_manager = None

    def __init__(self, db_path: str = "alerts.db"):
        """
        Initialize the database

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    @classmethod
    def get_class_manager(cls) -> ClassManager:
        """Get class manager instance, creating it if needed"""
        if cls._class_manager is None:
            cls._class_manager = ClassManager()
        return cls._class_manager

    def _init_db(self):
        """Initialize database schema if not exists"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    roi_name TEXT,
                    roi_index INTEGER,
                    alert_message TEXT,
                    snapshot_path TEXT,
                    video_path TEXT,
                    severity INTEGER DEFAULT 2,
                    camera_id TEXT DEFAULT 'main'
                )
            """)

            # Create detections table for storing individual detections in alerts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id INTEGER,
                    class_id INTEGER,
                    class_name TEXT,
                    confidence REAL,
                    x1 REAL,
                    y1 REAL,
                    x2 REAL,
                    y2 REAL,
                    FOREIGN KEY(alert_id) REFERENCES alerts(id) ON DELETE CASCADE
                )
            """)

            # Create statistics table for aggregated data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    hour INTEGER,
                    camera_id TEXT,
                    roi_id INTEGER,
                    class_id INTEGER,
                    count INTEGER
                )
            """)

            # Enable foreign keys
            cursor.execute("PRAGMA foreign_keys = ON")

            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def insert_alert(self, timestamp: str, roi_name: str, roi_index: int,
                     alert_message: str, snapshot_path: str, video_path: str,
                     severity: int = 2, camera_id: str = "main") -> int:
        """
        Insert a new alert into the database

        Args:
            timestamp: Alert timestamp string
            roi_name: Name of the ROI
            roi_index: Index of the ROI
            alert_message: Alert message (usually JSON string of class counts)
            snapshot_path: Path to snapshot image
            video_path: Path to video clip
            severity: Alert severity (1=low, 2=medium, 3=high)
            camera_id: Camera identifier

        Returns:
            ID of the inserted alert
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO alerts (
                    timestamp, roi_name, roi_index, alert_message, 
                    snapshot_path, video_path, severity, camera_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, roi_name, roi_index, alert_message,
                  snapshot_path, video_path, severity, camera_id))

            alert_id = cursor.lastrowid

            # Parse alert_message (JSON) and insert individual detections
            try:
                class_counts = json.loads(alert_message)

                # Load class names from class manager
                class_names = self.get_class_manager().get_class_names()

                # Insert into statistics table
                alert_date = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                alert_day = alert_date.strftime("%Y-%m-%d")
                alert_hour = alert_date.hour

                for class_id, count in class_counts.items():
                    class_id = int(class_id)
                    class_name = class_names.get(class_id, f"Unknown-{class_id}")

                    # Add to statistics
                    self._update_statistics(alert_day, alert_hour, camera_id,
                                            roi_index, class_id, count, cursor)
            except Exception as e:
                logger.error(f"Error processing alert detections: {e}")

            conn.commit()
            conn.close()

            logger.info(f"Alert inserted with ID {alert_id}")
            return alert_id
        except Exception as e:
            logger.error(f"Error inserting alert: {e}")
            return -1

    def _update_statistics(self, date: str, hour: int, camera_id: str,
                           roi_id: int, class_id: int, count: int, cursor):
        """Update the statistics table with detection counts"""
        try:
            # Check if entry exists
            cursor.execute("""
                SELECT id, count FROM statistics
                WHERE date = ? AND hour = ? AND camera_id = ? 
                AND roi_id = ? AND class_id = ?
            """, (date, hour, camera_id, roi_id, class_id))

            result = cursor.fetchone()

            if result:
                # Update existing entry
                stat_id, current_count = result
                new_count = current_count + count

                cursor.execute("""
                    UPDATE statistics SET count = ?
                    WHERE id = ?
                """, (new_count, stat_id))
            else:
                # Insert new entry
                cursor.execute("""
                    INSERT INTO statistics (date, hour, camera_id, roi_id, class_id, count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (date, hour, camera_id, roi_id, class_id, count))
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    def get_alerts(self, limit: int = 100, offset: int = 0,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   roi_index: Optional[int] = None,
                   camera_id: Optional[str] = None,
                   severity: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get alerts with optional filtering

        Args:
            limit: Maximum number of alerts to return
            offset: Offset for pagination
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            roi_index: Filter by ROI index
            camera_id: Filter by camera ID
            severity: Filter by severity level

        Returns:
            List of alert dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()

            query = "SELECT * FROM alerts WHERE 1=1"
            params = []

            # Add filters
            if start_date:
                query += " AND timestamp >= ?"
                params.append(f"{start_date} 00:00:00")

            if end_date:
                query += " AND timestamp <= ?"
                params.append(f"{end_date} 23:59:59")

            if roi_index is not None:
                query += " AND roi_index = ?"
                params.append(roi_index)

            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)

            if severity:
                query += " AND severity = ?"
                params.append(severity)

            # Add order by and limit
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to list of dictionaries
            alerts = [dict(row) for row in rows]

            # Parse alert_message JSON for each alert
            for alert in alerts:
                try:
                    alert["class_counts"] = json.loads(alert["alert_message"])
                except:
                    alert["class_counts"] = {}

            conn.close()
            return alerts
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []

    def get_alert_count(self) -> int:
        """Get total number of alerts in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM alerts")
            count = cursor.fetchone()[0]

            conn.close()
            return count
        except Exception as e:
            logger.error(f"Error getting alert count: {e}")
            return 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from the database

        Returns:
            Dictionary with various statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get total alert count
            cursor.execute("SELECT COUNT(*) FROM alerts")
            total_alerts = cursor.fetchone()[0]

            # Get alerts by severity
            cursor.execute("""
                SELECT severity, COUNT(*) as count
                FROM alerts
                GROUP BY severity
                ORDER BY severity
            """)
            severity_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Get alerts by camera
            cursor.execute("""
                SELECT camera_id, COUNT(*) as count
                FROM alerts
                GROUP BY camera_id
                ORDER BY count DESC
            """)
            camera_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Get alerts by ROI
            cursor.execute("""
                SELECT roi_name, COUNT(*) as count
                FROM alerts
                GROUP BY roi_name
                ORDER BY count DESC
            """)
            roi_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Get alerts by day (last 30 days)
            cursor.execute("""
                SELECT strftime('%Y-%m-%d', timestamp) as day, COUNT(*) as count
                FROM alerts
                WHERE timestamp >= date('now', '-30 days')
                GROUP BY day
                ORDER BY day
            """)
            day_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Get most common classes
            cursor.execute("""
                SELECT class_id, SUM(count) as total
                FROM statistics
                GROUP BY class_id
                ORDER BY total DESC
                LIMIT 10
            """)

            class_counts = {}
            # Get class names from class manager
            class_names = self.get_class_manager().get_class_names()

            for row in cursor.fetchall():
                class_id = row[0]
                class_name = class_names.get(class_id, f"Unknown-{class_id}")
                class_counts[class_name] = row[1]

            conn.close()

            return {
                "total_alerts": total_alerts,
                "alerts_by_severity": severity_counts,
                "alerts_by_camera": camera_counts,
                "alerts_by_roi": roi_counts,
                "alerts_by_day": day_counts,
                "top_classes": class_counts
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                "total_alerts": 0,
                "alerts_by_severity": {},
                "alerts_by_camera": {},
                "alerts_by_roi": {},
                "alerts_by_day": {},
                "top_classes": {}
            }

    def clear_all_alerts(self) -> bool:
        """
        Clear all alerts from the database

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM alerts")
            cursor.execute("DELETE FROM detections")
            cursor.execute("DELETE FROM statistics")

            conn.commit()
            conn.close()

            logger.info("All alerts cleared from database")
            return True
        except Exception as e:
            logger.error(f"Error clearing alerts: {e}")
            return False

    def export_to_csv(self, file_path: str) -> bool:
        """
        Export alerts to CSV file

        Args:
            file_path: Path to save CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all alerts
            cursor.execute("SELECT * FROM alerts")
            rows = cursor.fetchall()

            # Get column names
            column_names = [description[0] for description in cursor.description]

            # Write to CSV
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(column_names)

                # Write data
                writer.writerows(rows)

            conn.close()

            logger.info(f"Exported {len(rows)} alerts to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def delete_alert(self, alert_id: int) -> bool:
        """
        Delete an alert from the database

        Args:
            alert_id: ID of the alert to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))

            conn.commit()
            conn.close()

            logger.info(f"Deleted alert {alert_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting alert: {e}")
            return False