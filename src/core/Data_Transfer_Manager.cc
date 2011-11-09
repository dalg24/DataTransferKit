//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   core/Data_Transfer_Manager.cc
 * \author Stuart Slattery
 * \date   Wed Oct 05 11:02:44 2011
 * \brief  Data_Transfer_Manager member definitons.
 */
//---------------------------------------------------------------------------//
// $Id: template.cc,v 1.3 2008/01/02 17:18:47 9te Exp $
//---------------------------------------------------------------------------//

#include "Data_Transfer_Manager.hh"
#include "harness/DBC.hh"

namespace coupler
{

//---------------------------------------------------------------------------//
// Constructor.
Data_Transfer_Manager::Data_Transfer_Manager(const Communicator &comm_global)
    : d_comm_global(comm_global)
{ /* ... */ }

//---------------------------------------------------------------------------//
// Destructor.
Data_Transfer_Manager::~Data_Transfer_Manager()
{ /* ... */ }

//---------------------------------------------------------------------------//
// Register a physics to be controlled by the manager.
void Data_Transfer_Manager::add_physics(const std::string &physics_name,
					Transfer_Evaluator<DataType> *te)
{
    // Make a physics object.
    SP_Physics new_physics = new Physics(te, d_comm_global);

    // Add it to the physics database.
    d_physics_db.insert( Physics_Pair(physics_name, new_physics) );
}

//---------------------------------------------------------------------------//
// Build the topology map for transfer from a source physics to a target
// physics for a particular field.
void Data_Transfer_Manager::map(const std::string &field_name,
				const std::string &source_physics,
				const std::string &target_physics);
{
    // Operate on the global communicator.
    nemesis::set_internal_comm(d_comm_global);

    // Get the physics that we are operating on.
    SP_Physics source = d_physics_db[source_physics];
    SP_Physics target = d_physics_db[target_physics];

    // Require that these physics support this field.
    Insist( source->field_supported(field_name) &&
	    target->field_supported(field_name) );

    // Create a mapper.
    Mapper mapper(d_comm_global, field_name, source, target);

    // Generate the map.
    mapper.map();
}

//---------------------------------------------------------------------------//
// Transfer data associated with a field from a source physics to a target
// physics. 
void Data_Transfer_Manager::transfer(const std::string &field_name,
				     const std::string &source_physics,
				     const std::string &target_physics)
{
    // Operate on the global communicator.
    nemesis::set_internal_comm(d_comm_global);

    // Get the physics we are operating on.
    SP_Physics source = d_physics_db[source_physics];
    SP_Physics target = d_physics_db[target_physics];

    // Require that these physics support this field.
    Insist( source->field_supported(field_name) &&
	    target->field_supported(field_name) );
    
    // Create a messenger.
    Messenger<DataType> msgr(d_comm_global, field_name, source, target);

    // Transfer the field.
    msgr.communicate();
}

//---------------------------------------------------------------------------//

} // end namespace coupler

//---------------------------------------------------------------------------//
//                 end of Data_Transfer_Manager.cc
//---------------------------------------------------------------------------//
