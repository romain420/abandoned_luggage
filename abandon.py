####################################################################################################################""""""
# how to implement the alert launcher:
#
# 1- create an instance of the object LanceurAlerte
#     alertLauncher = LanceurAlerte(False)
#     Si False, aucun output n'est effectué (à part les alertes), si True, tout est print (dont les alertes)
#
# 2- lancer l'analyse des outputs à chaque frame, par la fonction suivante
#     alertLauncher.analyse_outputs(outputs)
####################################################################################################################

import numpy as np

def add_center(outputs):

  if not isinstance(outputs, list):
    result = np.zeros(shape=(outputs.shape[0], outputs.shape[1]+2))
    for index in range(outputs.shape[0]):
      result[index] = np.append(outputs[index], (outputs[index][1] - outputs[index][0], outputs[index][2] - outputs[index][3]))
    return result
  return outputs


def prepare_output(track):

# transform track to numpy array, in the right format, also add center
  bbox = track.to_tlbr()
  class_name = track.get_class()
  if class_name == 'baggage' or class_name == 'handbag' or class_name == 'suitcase' or class_name == 'backpack':
    class_id = 0
  else: class_id = 1
  track_id = track.track_id
  output = np.array((int(bbox[0]),int(bbox[2]), int(bbox[1]), int(bbox[3]), track_id, class_id, int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])))
  return output


def dist_two_points( a_x, a_y, b_x, b_y):

    return np.sqrt( (a_y - b_y)**2 + (a_x - b_x)**2 )


class Entite:
  
  def __init__(self):
    self.composantes = []
    self.frame_immobile = 0
    self.abandon = False

  def __init__(self, pred):
    self.composantes = pred
    self.frame_immobile = 0
    self.abandon = False

  def get_composantes(self): return self.composantes

  def get_frame_immobile(self): return self.frame_immobile

class LanceurAlerte:

  def __init__(self, printer):
    self.listeEntite = []
    self.printer = printer

  def get_listeEntite(self):
    return self.listeEntite

  def ajout_entite(self, entite):
    self.listeEntite.append(entite)

  def ajout_pred(self, output):
    entite = Entite(output)
    self.ajout_entite(entite)

  def pred_dans_liste(self, output):
    if output[5] == 1: # 0 => luggage,    1 => person,    on ne veut que les luggage   maybe float, a voir si ca pose pb
      return -1
    else:
      for index, entite in enumerate(self.listeEntite):
        if entite.composantes[4] == output[4] and entite.composantes[5] == output[5]:
          return index # la prediction a le meme id et la meme classe qu'une entite de la liste
    return None # aucune des entites de la liste n'est la meme que la prediction

  def distance_frame_precedente(self, output, index):
    centre_a = (self.listeEntite[index].composantes[6], self.listeEntite[index].composantes[7])
    centre_b = (output[6], output[7])
    return dist_two_points(centre_a[0], centre_a[1], centre_b[0], centre_b[1])

  def update_entite(self, output, index, distance):
    self.listeEntite[index].composantes = output
    if distance < 30:
      self.listeEntite[index].frame_immobile += 1
    else:
      self.listeEntite[index].frame_immobile = 0

  def analyse_outputs(self, outputs):
    if self.printer: print("\n\ndebut analyse des predictions")
    outputs = add_center(outputs)
    for output in outputs:
      self.analyse_entite_v5(output)
    self.analyse_liste()

  def analyse_entite_v5(self, output):
    index = self.pred_dans_liste(output)
    if index == -1:
      if self.printer: print("la prediction est une personne, skip")
      return
    if index == None:
      self.ajout_pred(output)
      if self.printer: print("adding prediction(luggage only) to entity_list")
    else:
      distance = self.distance_frame_precedente(output, index)
      if self.printer: print("distance = ", distance, "  updating entity")
      self.update_entite(output, index, distance) 

  def analyse_entite_v4(self, track):
    output = prepare_output(track)
    index = self.pred_dans_liste(output)
    if index == -1:
      if self.printer: print("la prediction est une personne, skip")
      return
    if index == None:
      self.ajout_pred(output)
      if self.printer: print("adding prediction(luggage only) to entity_list")
    else:
      distance = self.distance_frame_precedente(output, index)
      if self.printer: print("distance = ", distance, "  updating entity")
      self.update_entite(output, index, distance)   

  def analyse_liste(self):
    if self.printer: print("\nanalyse de la liste d'entite ...")
    for entite in self.listeEntite:
      if entite.frame_immobile >= 30:
        entite.abandon = True
        print("/!\/!\/!\/!\/!\   ALERTE ABANDON    /!\/!\/!\/!\/!\ ")
        print("L'entite numero ", entite.composantes[4], " avec la classe numero ", entite.composantes[5], "  est considere comme ABANDONNE! Verifiez la situation")
        print("/!\/!\/!\/!\/!\   ALERTE ABANDON    /!\/!\/!\/!\/!\ ")
      else:
        entite.abandon = False
    if self.printer: print("Done\n")