/******************************************************************************
create_scans.cpp
Use the stage simulator to generate a dataset of lidar scans.
*******************************************************************************
The MIT License (MIT)

  Copyright (c) 2016 Matthew A. Klein

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
******************************************************************************/

// libstage
#include <stage.hh>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath> // for cos and sin
#include <cstring> // for strcmp()
#include <unistd.h> // For usleep() - to be removed.

#include "cnpy.h" // let's see if it works

class CreateScans
{
private:
    // File to store scans. It is opened in the contructor, written to in
    // WorldCallback(), and closed in the destructor.
    std::ofstream scanfile;

    // The models that we're interested in
    std::vector<Stg::ModelRanger *> lasermodels;
    std::vector<Stg::ModelPosition *> positionmodels;

    // The shapes that we detect with the lidar
    std::vector<Stg::Model *> shapemodels;

    //a structure representing a robot inthe simulator
    struct StageRobot
    {
        //stage related models
        //one position
        Stg::ModelPosition* positionmodel;
        //multiple rangers per position
        std::vector<Stg::ModelRanger *> lasermodels;
    };

    std::vector<StageRobot const *> robotmodels_;

    // Current values for shape, size, r, theta, rot. They are set in
    // generate_scans() and written to file in WorldCallback().
    int shapeCurr;
    double sizeCurr;
    double rCurr;
    double thetaCurr;
    double rotCurr;

    // A helper function that is executed for each stage model.  We use it
    // to search for models of interest.
    static void ghfunc(Stg::Model* mod, CreateScans* node);

    static bool s_update(Stg::World* world, CreateScans* node)
    {
        node->WorldCallback();
        // We return false to indicate that we want to be called again (an
        // odd convention, but that's the way that Stage works).
        return false;
    }

public:
    // A friend function of block that can alter the private `pts` member
    // variable. This is used to change verticies of the block.
    void reassign_verticies(Stg::Model* mod, std::vector<double>& x,
                                   std::vector<double>& y);

    /* move_shape() moves the shape model to an x,y position with rotation
       equal to a. a should be equal to zero when the shape's vertical
       direction points in the same direction as r. The parameter rot then
       rotates the shape out of line with r. All angles are in radians.
    */
    void move_shape(Stg::Model* mod, double r, double theta, double rot);

    // Subscribe to models of interest.  Currently, we find and subscribe
    // to the first 'laser' model and the first 'position' model.  Returns
    // 0 on success (both models subscribed), -1 otherwise.

    // Cycles through shapes, location, rotation, etc, and calls UpdateWorld(),
    // which triggers WorldCallback(), which writes scans to disk.
    void generate_scans();

    int SubscribeModels();

    // Our callback
    void WorldCallback();

    // Do one update of the world.  May pause if the next update time
    // has not yet arrived.
    bool UpdateWorld();

    CreateScans(int argc, char** argv, const char* fname);
    ~CreateScans();
    Stg::World* world;
};

void
CreateScans::reassign_verticies(Stg::Model* mod, std::vector<double>& x,
                                std::vector<double>& y)
{
    // Do nothing.
}

/* move_shape() moves the shape model to an x,y position with rotation equal to
   a. a should be equal to zero when the shape's vertical direction points in
   the same direction as r. The parameter rot then rotates the shape out of line
   with r. All angles are in radians.
 */
void
CreateScans::move_shape(Stg::Model* mod, double r, double theta, double rot)
{
    Stg::Pose pose;
    pose.x = r * cos(theta);
    pose.y = r * sin(theta);
    pose.z = 0;
    // shape is originially in y-direction, rotate -90 degrees first
    pose.a = -M_PI/2 + theta + rot;
    mod->SetPose(pose);
}

void
CreateScans::generate_scans()
{
    // Parameter sweep
    int shape; // shape
    int shapeMin = 0;
    int shapeMax = this->shapemodels.size()-1;
    int shapeNum = this->shapemodels.size();
    double r; // distance from lidar
    double rMin = 5;
    double rMax = 25;
    int rNum = 5;
    double theta; // angle from lidar (0 = straight ahead)
    double thetaMin = -M_PI_2;
    double thetaMax = M_PI_2;
    int thetaNum = 10;
    double rot; // shape's rotation from r (0 lies along r)
    double rotMin = -M_PI;
    double rotMax = M_PI;
    int rotNum = 5;
    double size; // multiplier of original size
    double sizeMin = 1;
    double sizeMax = 5;
    int sizeNum = 5;

    // Display the number of scans recorded
    int num_scans = shapeNum*rNum*thetaNum*rotNum*sizeNum;
    std::cout << "Number of Scans : " << num_scans << std::endl;
    int count = 1;
    // Cycle through the shapes
    for (int m = 0; m < shapeNum; m++) {
        // Read shape .csv file and set verticies.
        this->shapeCurr = m;
        // Get original shape size
        Stg::Geom geom = this->shapemodels[m]->GetGeom();
        double xOrig = geom.size.x;
        double yOrig = geom.size.y;
        // Cycle through the sizes
        for (int l = 0; l < sizeNum; l++) {
            size = (sizeMin + (l/(double)(sizeNum-1))*(sizeMax-sizeMin));
            geom.size.x = xOrig * size;
            geom.size.y = yOrig * size;
            this->sizeCurr = size;
            // And update the size
            this->shapemodels[0]->SetGeom(geom);
            // Cycle through the radii
            for (int i = 0; i < rNum; i++) {
                r = rMin + (i/(double)(rNum-1))*(rMax-rMin);
                this->rCurr = r;
                // Cycle through the angles
                for (int j = 0; j < thetaNum ; j++) {
                    theta = thetaMin
                        + (j/(double)(thetaNum-1))*(thetaMax-thetaMin);
                    this->thetaCurr = theta;
                    // Cycle through the rotations
                    for (int k = 0; k < rotNum; k++) {
                        rot = rotMin + (k/(double)(rotNum-1))*(rotMax-rotMin);
                        this->rotCurr = rot;
                        // Then move the model
                        this->move_shape(this->shapemodels[m], r, theta, rot);
                        // write the scan header to the file
                        // block type, size, distance, angle, rotation,
                        // noise level (always zero here),
                        this->scanfile << m << "," << size << "," << r << ","
                                       << theta << "," << rot << "," << 0
                                       << ",";
                        // and update the world (scan is appended to line)
                        this->UpdateWorld();
                        // DEBUG
                        std::cout << "Percent Done = "
                                  << (double)count/(double)num_scans
                                  << "%" << std::endl;
                        count++;
                    }
                }
            }
        }
        // Move the shape back behind the lidar when we're done with it
        this->move_shape(this->shapemodels[m], -5, 0, 0);
    }
}

void
CreateScans::ghfunc(Stg::Model* mod, CreateScans* node)
{
    if (dynamic_cast<Stg::ModelRanger *>(mod))
        node->lasermodels.push_back(dynamic_cast<Stg::ModelRanger *>(mod));
    if (dynamic_cast<Stg::ModelPosition *>(mod)) {
        Stg::ModelPosition * p = dynamic_cast<Stg::ModelPosition *>(mod);
        node->positionmodels.push_back(p);
    }
    if (strcmp("shape",mod->Token()) <= 0 ) {
        node->shapemodels.push_back(dynamic_cast<Stg::Model *>(mod));
    }
}

// Subscribe to models of interest.  Currently, we find and subscribe
// to the first 'laser' model and the first 'position' model.  Returns
// 0 on success (both models subscribed), -1 otherwise.
int
CreateScans::SubscribeModels()
{
    for (size_t r = 0; r < this->positionmodels.size(); r++)
    {
        StageRobot* new_robot = new StageRobot;
        new_robot->positionmodel = this->positionmodels[r];
        new_robot->positionmodel->Subscribe();


        for (size_t s = 0; s < this->lasermodels.size(); s++)
        {
            if (this->lasermodels[s] and this->lasermodels[s]->Parent() == new_robot->positionmodel)
            {
                new_robot->lasermodels.push_back(this->lasermodels[s]);
                this->lasermodels[s]->Subscribe();
                std::cout << "Laser added to robot" << std::endl;
            }
        }
        this->robotmodels_.push_back(new_robot);
    }
    return(0);
}

bool
CreateScans::UpdateWorld()
{
    return this->world->UpdateAll();
}

void
CreateScans::WorldCallback()
{
    double* input = new double[181];
    double* target = new double[6];

    const unsigned int targetShape[] = {1,6};
    const unsigned int inputShape[] = {1,181};

    //loop on the robot models
    for (size_t r = 0; r < this->robotmodels_.size(); ++r)
    {
        StageRobot const * robotmodel = this->robotmodels_[r];

        //loop on the laser devices for the current robot
        for (size_t s = 0; s < robotmodel->lasermodels.size(); ++s)
        {
            Stg::ModelRanger const* lasermodel = robotmodel->lasermodels[s];
            const std::vector<Stg::ModelRanger::Sensor>& sensors
                = lasermodel->GetSensors();

            // for now we access only the zeroth sensor of the ranger - good
            // enough for most laser models that have a single beam origin
            const Stg::ModelRanger::Sensor& sensor = sensors[0];

            if( sensor.ranges.size() )
            {
                for(unsigned int i = 0; i < sensor.ranges.size(); i++)
                {
                    input[i] = sensor.ranges[i];
                    // Write each range to file
                    // TODO This should be done in reverse order
                    this->scanfile << sensor.ranges[i];
                    // Add a comma between all ranges in a scan except the last
                    if (i < sensor.ranges.size()-1)
                        this->scanfile << ",";
                }
                // Append a newline at the end of each scan
                this->scanfile << "\n";

                // Append targets and scans to .npy file
                // cnpy::npy_save("targets.npy",target,targetShape,2,"a");
                cnpy::npy_save("scans.npy",input,inputShape,2,"a");
            }
        }
    }
    delete[] input;
}

CreateScans::CreateScans(int argc, char** argv, const char* fname)
{
    // initialize libstage
    Stg::Init( &argc, &argv );

    // create a new World or WorldGui object
    if(false) // don't use a gui (set true if a gui is desired)
        this->world = new Stg::WorldGui(600, 400, "Stage (Create Scans)");
    else
        this->world = new Stg::World();

    // and load the worldfile
    this->world->Load(fname);

    // We add our callback here, after the Update, so avoid our callback
    // being invoked before we're ready.
    this->world->AddUpdateCallback((Stg::world_callback_t)s_update, this);
    this->world->ForEachDescendant((Stg::model_callback_t)ghfunc, this);
    std::cout << "shapemodels = " << this->shapemodels.size() << std::endl; //DEBUG
    // Open up the file to write to
    scanfile.open("scans.csv");
}

CreateScans::~CreateScans()
{
    for (std::vector<StageRobot const*>::iterator r = this->robotmodels_.begin(); r != this->robotmodels_.end(); ++r)
        delete *r;
    for (std::vector<Stg::Model*>::iterator r = this->shapemodels.begin(); r != this->shapemodels.end(); ++r)
        delete *r;

    // Close the file
    this->scanfile.close();
}

int main(int argc, char *argv[])
{
  std::cout << "It's alive!" << std::endl;

  // Instantiate a CreateScans object
  CreateScans cs(argc-1,argv,argv[argc-1]);

  if(cs.SubscribeModels() != 0)
      exit(-1);

  // New in Stage 4.1.1: must Start() the world.
  cs.world->Start();
  cs.generate_scans();

  return 0;
}
