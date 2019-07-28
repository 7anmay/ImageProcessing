int i;
int motorPin1 = 3;
int motorPin2 = 6;
int motorPin3 = 5;
int motorPin4 = 9;
int ledpin = 4;
char t;

void setup() {
  Serial.begin(9600);
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  pinMode(motorPin3, OUTPUT);
  pinMode(motorPin4, OUTPUT);
  pinMode(ledpin ,OUTPUT);
  digitalWrite(ledpin ,LOW);
  }

void loop() {
if (Serial.available() > 0)
  {  t = Serial.read();
  if (t == '1')
  {// blink once
    
      digitalWrite(ledpin , HIGH);
      delay(250);
      digitalWrite(ledpin , LOW); 
      delay(250);
      }
  else if (t == '2')
  {// blink twice
    for(i=0;i<2;i++)
    {
      digitalWrite(ledpin , HIGH);
      delay(250);
      digitalWrite(ledpin , LOW); 
      delay(250);
    }
  }
   else if (t == '3')
  {// blink thrice
    for(i=0;i<3;i++)
    {
      digitalWrite(ledpin , HIGH);
      delay(250);
      digitalWrite(ledpin , LOW); 
      delay(250); 
      }
  }
   else if (t == '4')
  {// blink four
    for(i=0;i<4;i++)
    {
      digitalWrite(ledpin , HIGH);
      delay(250);
      digitalWrite(ledpin , LOW); 
      delay(250);
      }
  }

  else if (t == '5')
  {// blink five
    for(i=0;i<5;i++)
    {
      digitalWrite(ledpin , HIGH);
      delay(250);
      digitalWrite(ledpin , LOW); 
      delay(250);
      }
  }
  
  else if (t == 'm')
  { 
    digitalWrite(ledpin,LOW);
    analogWrite(motorPin1, 200);
    analogWrite(motorPin2, 0);
    analogWrite(motorPin3, 200);
    analogWrite(motorPin4, 0);
  }
  else if ( t == 'p')
  { //anticlockwise
    digitalWrite(ledpin,LOW);
    analogWrite(motorPin1, 75);
    analogWrite(motorPin2, 0);  
    analogWrite(motorPin3, 0);
    analogWrite(motorPin4, 75);
    
  }
  else if ( t == 'o')
  {//clockwise
    digitalWrite(ledpin,LOW);
    analogWrite(motorPin1, 0);
    analogWrite(motorPin2, 75);
    analogWrite(motorPin3, 75);
    analogWrite(motorPin4, 0);
  }
  else if ( t == 's')
  {
    digitalWrite(ledpin,LOW);
    analogWrite(motorPin1, 0);
    analogWrite(motorPin2, 0);
    analogWrite(motorPin3, 0);
    analogWrite(motorPin4, 0);
  }
 }
}

