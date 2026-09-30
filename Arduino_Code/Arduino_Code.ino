
#include <Servo.h>
Servo servoVer; //Vertical Servo
Servo servoHor; //Horizontal Servo
Servo crawler;
int x;
int y;
int z;
int prevX;
int prevY;
int prevZ;
void setup()
{
  Serial.begin(9600);
  servoVer.attach(10); //Attach Vertical Servo to Pin 10
  servoHor.attach(9);
  crawler.attach(6);
  servoVer.write(0);
  servoHor.write(90);
  crawler.write(90);
}
void Pos()
{
  if(prevX != x || prevY != y || prevZ != z)
  {
    int servoX = map(x, 10, 600, 0, 180);
    int servoY = map(y, 10, 300, 0, 60);
    int servoZ = map(z, 10, 300, 0, 90);

    servoX = constrain(servoX, 0, 179);
    servoY = constrain(servoY, 0, 60);
    servoZ = constrain(servoZ, 0, 90);

    servoHor.write(servoX);
    servoVer.write(servoY);
    crawler.write(servoZ);

    prevX = x;
    prevY = y;
    prevZ = z;
  }
}
void loop()
{
  if(Serial.available() > 0)
  {
    if(Serial.read() == 'X')
    {
      x = Serial.parseInt();
      if(Serial.read() == 'Y')
      {
        y = Serial.parseInt();
        if(Serial.read() == 'Z')
        {
          z = Serial.parseInt();
          Pos();
        }
      }
      while(Serial.available() > 0)
      {
        Serial.read();
      }
    }
  }
}
