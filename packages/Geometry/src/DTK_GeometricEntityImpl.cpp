//---------------------------------------------------------------------------//
/*
  Copyright (c) 2014, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the Oak Ridge National Laboratory nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//---------------------------------------------------------------------------//
/*!
 * \brief DTK_GeometricEntityImpl.cpp
 * \author Stuart R. Slattery
 * \brief Geometric entity implementation.
 */
//---------------------------------------------------------------------------//

#include "DTK_GeometricEntityImpl.hpp"
#include "DTK_DBC.hpp"

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
// Constructor.
GeometricEntityImpl::GeometricEntityImpl()
{ /* ... */ }

//---------------------------------------------------------------------------//
//brief Destructor.
GeometricEntityImpl::~GeometricEntityImpl()
{ /* ... */ }

//---------------------------------------------------------------------------//
// Return a string indicating the derived entity type.
std::string GeometricEntityImpl::entityType() const
{
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
    return "Not Implemented";
}

//---------------------------------------------------------------------------//
// Get the unique global identifier for the entity.
EntityId GeometricEntityImpl::id() const
{ 
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
    return -1;
}
    
//---------------------------------------------------------------------------//
// Get the parallel rank that owns the entity.
int GeometricEntityImpl::ownerRank() const
{ 
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
    return -1;
}
//---------------------------------------------------------------------------//
// Return the physical dimension of the entity.
int GeometricEntityImpl::physicalDimension() const
{ 
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
    return -1;
}

//---------------------------------------------------------------------------//
// Return the parametric dimension of the entity.
int GeometricEntityImpl::parametricDimension() const
{ 
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
    return -1;
}

//---------------------------------------------------------------------------//
// Return the entity measure (volume for a 3D entity, area for 2D, and length
// for 1D). 
double GeometricEntityImpl::measure() const
{ 
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
    return -1.0;
}

//---------------------------------------------------------------------------//
// Return the centroid of the entity.
void GeometricEntityImpl::centroid( 
    Teuchos::ArrayView<const double>& centroid ) const
{ 
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
}

//---------------------------------------------------------------------------//
// Return the axis-aligned bounding box around the entity.
void GeometricEntityImpl::boundingBox( Teuchos::Tuple<double,6>& bounds ) const
{ 
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
}

//---------------------------------------------------------------------------//
// Perform a safeguard check for mapping a point to the reference space of an
// entity using the given tolerance. 
void GeometricEntityImpl::safeguardMapToReferenceFrame(
    const Teuchos::ParameterList& parameters,
    const Teuchos::ArrayView<const double>& point,
    MappingStatus& status ) const
{ 
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
}

//---------------------------------------------------------------------------//
// Map a point to the reference space of an entity. Return the
// parameterized point. 
void GeometricEntityImpl::mapToReferenceFrame( 
    const Teuchos::ParameterList& parameters,
    const Teuchos::ArrayView<const double>& point,
    const Teuchos::ArrayView<double>& reference_point,
    MappingStatus& status ) const
{ 
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
}

//---------------------------------------------------------------------------//
// Determine if a reference point is in the parameterized space of an entity.
bool GeometricEntityImpl::checkPointInclusion( 
    const Teuchos::ParameterList& parameters,
    const Teuchos::ArrayView<const double>& reference_point ) const
{ 
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
    return false;
}

//---------------------------------------------------------------------------//
// Map a reference point to the physical space of an entity.
void GeometricEntityImpl::mapToPhysicalFrame( 
    const Teuchos::ArrayView<const double>& reference_point,
    const Teuchos::ArrayView<double>& point ) const
{
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
}

//---------------------------------------------------------------------------//
// Return a string indicating the derived object type.
std::string GeometricEntityImpl::objectType() const
{
    return entityType();
}

//---------------------------------------------------------------------------//
// Serialize the entity into a buffer.
void GeometricEntityImpl::serialize( 
    const Teuchos::ArrayView<char>& buffer ) const
{
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
}

//---------------------------------------------------------------------------//
// Deserialize an entity from a buffer.
void GeometricEntityImpl::deserialize( 
    const Teuchos::ArrayView<const char>& buffer )
{
    DTK_REMEMBER( bool not_implemented = true );
    DTK_INSIST( !not_implemented );
}

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//
// end DTK_GeometricEntityImpl.cpp
//---------------------------------------------------------------------------//
