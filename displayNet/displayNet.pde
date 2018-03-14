PImage img=createImage(600,600,RGB);
PrintWriter save;

void setup(){
  size(600,600);
  background(255);
  fill(0);
  noStroke();
  save=createWriter("pics.txt");
}

void draw(){
  if(mousePressed){
    ellipse(mouseX, mouseY, 50, 50);
  }
}

void keyPressed(){
  if(key==' '){
    img=get();
    img.resize(28,28);
    img.loadPixels();
    for(int i=0; i<28*28; i++){
      color c=img.pixels[i];
      save.println(int(((red(c)+green(c))+blue(c))/3));
    }
    background(255);
  }
  if(key=='x'){
    save.flush();
    save.close();
    exit();
  }
  //if(key==' '){
  //  loadPixels();
  //  img.loadPixels();
  //  for(int i=0; i<width*height; i++){
  //    img.pixels[i]=pixels[i];
  //  }
  //  img.updatePixels();
  //  updatePixels();
  //}
}