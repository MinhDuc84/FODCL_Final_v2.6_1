import logging
import smtplib
import os
import time
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import List, Optional, Any, Union

from notifications.base import BaseNotifier

logger = logging.getLogger("FOD.Notifications.Email")


class EmailNotifier(BaseNotifier):
    """
    Send notifications via email
    """

    def __init__(self, smtp_server: str, smtp_port: int,
                 username: str, password: str,
                 from_addr: str, to_addrs: Union[str, List[str]],
                 use_ssl: bool = True,
                 min_severity: int = 2,
                 max_retries: int = 3,
                 retry_delay: int = 5):
        """
        Initialize the email notifier

        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_addr: From email address
            to_addrs: To email address(es)
            use_ssl: Whether to use SSL/TLS
            min_severity: Minimum severity level to trigger notifications (1=low, 2=medium, 3=high)
            max_retries: Maximum number of retries for failed messages
            retry_delay: Delay between retries in seconds
        """
        super().__init__(name="Email", min_severity=min_severity)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr

        # Convert to_addrs to list if it's a string
        if isinstance(to_addrs, str):
            self.to_addrs = [to_addrs]
        else:
            self.to_addrs = to_addrs

        self.use_ssl = use_ssl
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def is_configured(self) -> bool:
        """Check if the notifier is properly configured"""
        return all([
            self.smtp_server,
            self.smtp_port,
            self.username,
            self.password,
            self.from_addr,
            self.to_addrs
        ])

    def send(self, alert: Any) -> bool:
        """
        Send alert notification via email

        Args:
            alert: The alert to send

        Returns:
            True if successful, False otherwise
        """
        if not self.is_configured():
            logger.error("Email notifier not properly configured")
            return False

        if alert.severity < self.min_severity:
            logger.debug(f"Alert severity {alert.severity} below minimum {self.min_severity}, skipping")
            return True  # Skip but return success

        # Create email message
        msg = MIMEMultipart()
        msg['From'] = self.from_addr
        msg['To'] = ', '.join(self.to_addrs)

        # Set subject based on severity
        severity_text = {
            1: "LOW",
            2: "MEDIUM",
            3: "HIGH"
        }.get(alert.severity, "UNKNOWN")

        msg['Subject'] = f"FOD Detection Alert - {severity_text} - ROI: {alert.roi_name}"

        # Format message body (both plain text and HTML)
        text_body = self.format_message(alert)
        html_body = self._format_html_message(alert)

        # Attach text and HTML parts
        msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        # Attach image if available
        if alert.snapshot_path and os.path.exists(alert.snapshot_path):
            try:
                with open(alert.snapshot_path, 'rb') as img_file:
                    img_data = img_file.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-ID', '<detection_image>')
                    msg.attach(image)
            except Exception as e:
                logger.error(f"Error attaching image to email: {e}")

        # Send email with retries
        for attempt in range(self.max_retries):
            try:
                # Create SMTP connection
                if self.use_ssl:
                    server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
                else:
                    server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                    server.starttls()

                # Login and send
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
                server.quit()

                logger.info(f"Successfully sent email alert")
                return True
            except Exception as e:
                logger.error(f"Error sending email (attempt {attempt + 1}/{self.max_retries}): {e}")

                # Delay before retry
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return False

    def _format_html_message(self, alert: Any) -> str:
        """
        Format alert into an HTML message

        Args:
            alert: The alert to format

        Returns:
            Formatted HTML message
        """
        # Format timestamp
        timestamp = alert.datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Get severity text and color
        severity_data = {
            1: ("LOW", "#3498db"),  # Blue
            2: ("MEDIUM", "#f39c12"),  # Orange
            3: ("HIGH", "#e74c3c")  # Red
        }.get(alert.severity, ("UNKNOWN", "#7f8c8d"))  # Gray

        severity_text, severity_color = severity_data

        # Get class names dictionary
        class_names = self._get_class_names()

        # Build detection table rows
        detection_rows = ""
        total_objects = 0

        for class_id, count in alert.class_counts.items():
            if count > 0:
                class_name = class_names.get(int(class_id), f"Unknown-{class_id}")
                detection_rows += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{count}</td>
                </tr>
                """
                total_objects += count

        # Add total row
        detection_rows += f"""
        <tr style="font-weight: bold; background-color: #f8f9fa;">
            <td>Total</td>
            <td>{total_objects}</td>
        </tr>
        """

        # Build HTML email
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: {severity_color}; color: white; padding: 10px; text-align: center; }}
                .content {{ padding: 20px; border: 1px solid #ddd; }}
                .info {{ margin-bottom: 20px; }}
                .info-row {{ margin-bottom: 10px; }}
                .info-label {{ font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .img-container {{ margin-top: 20px; text-align: center; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>FOD Detection Alert - {severity_text}</h2>
                </div>
                <div class="content">
                    <div class="info">
                        <div class="info-row">
                            <span class="info-label">ROI:</span> {alert.roi_name}
                        </div>
                        <div class="info-row">
                            <span class="info-label">Camera:</span> {alert.camera_id}
                        </div>
                        <div class="info-row">
                            <span class="info-label">Time:</span> {timestamp}
                        </div>
                    </div>

                    <h3>Detections</h3>
                    <table>
                        <tr>
                            <th>Object Type</th>
                            <th>Count</th>
                        </tr>
                        {detection_rows}
                    </table>

                    <div class="img-container">
                        <h3>Detection Image</h3>
                        <img src="cid:detection_image" alt="Detection Image">
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html