// FlyBrain Robot Bridge receiver for M5Stack Atom Matrix (ESP32-PICO-D4).
// Two continuous-rotation servos, onboard MPU6886 IMU, UDP JSON protocol v1.
//
// Safety model:
//   * Motors stay at neutral until MOTORS_CALIBRATED is true (compile time) AND
//     the front button has armed the robot (run time). Press again to disarm.
//   * Any of: no command for WATCHDOG_MS, Wi-Fi loss, IMU failure, emergency_stop
//     -> neutral pulses immediately.
//   * LED matrix: red = disarmed, green = armed, blue = waiting for Wi-Fi.
// Compiles with arduino-cli (see README). Not yet verified on a physical robot.

#include <ArduinoJson.h>
#include <M5Unified.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <math.h>

// ---- Local configuration --------------------------------------------------
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
IPAddress PC_IP(192, 168, 1, 10);
const uint16_t ROBOT_PORT = 9000, PC_PORT = 9001;

// Servo signal pins. G22 and G19 are exposed on the Atom Matrix bottom header;
// check your board revision. Power servos from a separate supply, common ground.
const int LEFT_SERVO_PIN = 22;
const int RIGHT_SERVO_PIN = 19;
// Calibrate with the robot lifted: neutral must not creep, +100 must drive forward.
const bool MOTORS_CALIBRATED = false;
const int LEFT_NEUTRAL_US = 1500, RIGHT_NEUTRAL_US = 1500;
const int SPAN_US = 400;                 // pulse offset at |speed| = 100
const bool LEFT_REVERSED = false;
const bool RIGHT_REVERSED = true;        // mirrored servo mounting
// ---------------------------------------------------------------------------

const uint32_t WATCHDOG_MS = 500;
const uint32_t TELEMETRY_MS = 50;
const int PWM_HZ = 50, PWM_BITS = 16;
const int LED_PIN = 27;

WiFiUDP udp;
uint32_t lastCommand = 0, lastTelemetry = 0;
long lastSequence = -1, telemetrySequence = 0;
float leftSpeed = 0, rightSpeed = 0;
bool haveCommand = false, armed = false;

uint32_t dutyFor(int pulseUs) {
  return (uint32_t)((double)pulseUs * ((1UL << PWM_BITS) - 1) * PWM_HZ / 1000000.0);
}

void writePulse(int pin, int channel, int pulseUs) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  (void)channel;
  ledcWrite(pin, dutyFor(pulseUs));
#else
  (void)pin;
  ledcWrite(channel, dutyFor(pulseUs));
#endif
}

int pulseFor(float speed, int neutral, bool reversed) {
  float s = constrain(speed, -100.0f, 100.0f) * (reversed ? -1 : 1);
  return neutral + (int)lround(s * SPAN_US / 100.0f);
}

void applyMotors(float left, float right) {
  bool move = MOTORS_CALIBRATED && armed;
  writePulse(LEFT_SERVO_PIN, 0, move ? pulseFor(left, LEFT_NEUTRAL_US, LEFT_REVERSED)
                                     : LEFT_NEUTRAL_US);
  writePulse(RIGHT_SERVO_PIN, 1, move ? pulseFor(right, RIGHT_NEUTRAL_US, RIGHT_REVERSED)
                                      : RIGHT_NEUTRAL_US);
}

void stopMotors() {
  leftSpeed = rightSpeed = 0;
  haveCommand = false;
  applyMotors(0, 0);
}

void showState() {
  static uint32_t shown = 0xFFFFFFFF;
  uint32_t color = WiFi.status() != WL_CONNECTED ? 0x000010 : (armed ? 0x001000 : 0x100000);
  if (color == shown) return;
  shown = color;
  // The ESP32 core's single-pixel RMT helper lights the first LED of the matrix,
  // dimly, which is enough for a status light and keeps current draw low.
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  rgbLedWrite(LED_PIN, (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF);
#else
  neopixelWrite(LED_PIN, (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF);
#endif
}

void setupHardware() {
  auto cfg = M5.config();
  M5.begin(cfg);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(LEFT_SERVO_PIN, PWM_HZ, PWM_BITS);
  ledcAttach(RIGHT_SERVO_PIN, PWM_HZ, PWM_BITS);
#else
  ledcSetup(0, PWM_HZ, PWM_BITS);
  ledcSetup(1, PWM_HZ, PWM_BITS);
  ledcAttachPin(LEFT_SERVO_PIN, 0);
  ledcAttachPin(RIGHT_SERVO_PIN, 1);
#endif
}

bool readIMU(float gyro[3], float accel[3]) {
  // M5Unified reports gyro in degrees/second and acceleration in g.
  if (!M5.Imu.isEnabled()) return false;
  M5.Imu.update();
  auto data = M5.Imu.getImuData();
  for (int i = 0; i < 3; ++i) {
    gyro[i] = data.gyro.value[i];
    accel[i] = data.accel.value[i];
  }
  return true;
}

bool validSpeed(JsonVariantConst value) {
  return !value.is<bool>() && (value.is<float>() || value.is<long>())
         && isfinite(value.as<float>()) && fabs(value.as<float>()) <= 100;
}

void handleCommand(const char* buffer, int count) {
  JsonDocument doc;
  if (deserializeJson(doc, buffer, count)) return;
  if (doc["type"] != "motor_command" || !doc["sequence"].is<long>()
      || doc["sequence"].is<bool>() || doc["sequence"].as<long>() < 0
      || !doc["emergency_stop"].is<bool>() || !validSpeed(doc["left"])
      || !validSpeed(doc["right"])) {
    return;
  }
  long sequence = doc["sequence"];
  if (doc["emergency_stop"].as<bool>()) stopMotors();
  if (sequence <= lastSequence) return;
  lastSequence = sequence;
  if (!doc["emergency_stop"].as<bool>()) {
    leftSpeed = doc["left"];
    rightSpeed = doc["right"];
    lastCommand = millis();
    haveCommand = true;
    applyMotors(leftSpeed, rightSpeed);
  }
}

void sendTelemetry() {
  float gyro[3], accel[3];
  if (!readIMU(gyro, accel)) {
    stopMotors();
    return;
  }
  for (int i = 0; i < 3; ++i) {
    if (!isfinite(gyro[i]) || !isfinite(accel[i])) {
      stopMotors();
      return;
    }
  }
  JsonDocument doc;
  doc["type"] = "telemetry";
  doc["sequence"] = telemetrySequence++;
  JsonArray g = doc["gyro"].to<JsonArray>();
  JsonArray a = doc["accel"].to<JsonArray>();
  for (int i = 0; i < 3; ++i) {
    g.add(gyro[i]);
    a.add(accel[i]);
  }
  doc["left_speed"] = leftSpeed;  // last command, not an encoder measurement
  doc["right_speed"] = rightSpeed;
  udp.beginPacket(PC_IP, PC_PORT);
  serializeJson(doc, udp);
  udp.endPacket();
}

void setup() {
  setupHardware();
  stopMotors();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    stopMotors();
    showState();
    delay(100);
  }
  udp.begin(ROBOT_PORT);
  showState();
}

void loop() {
  M5.update();
  if (M5.BtnA.wasPressed()) {
    armed = !armed;
    stopMotors();
    showState();
  }
  if (!haveCommand || millis() - lastCommand >= WATCHDOG_MS || WiFi.status() != WL_CONNECTED) {
    stopMotors();
  }
  int size = udp.parsePacket();
  if (size > 0) {
    char buffer[512];
    if (size >= (int)sizeof(buffer) || udp.remoteIP() != PC_IP) {
      while (udp.available()) udp.read();
    } else {
      handleCommand(buffer, udp.read(buffer, sizeof(buffer)));
    }
  }
  if (millis() - lastTelemetry >= TELEMETRY_MS) {
    lastTelemetry = millis();
    sendTelemetry();
  }
  showState();
}
